# Databricks notebook source
# MAGIC %md
# MAGIC # LangGraph tool-calling agent on the custom endpoint
# MAGIC
# MAGIC This notebook authors a LangGraph agent that uses `ChatDatabricks` pointed at a
# MAGIC **custom serving endpoint deployed by this project** (any endpoint serving vLLM
# MAGIC with native OpenAI tool calling via `--tool-call-parser gemma4`). The endpoint is
# MAGIC selected with the `ENDPOINT_NAME` widget / job parameter.
# MAGIC
# MAGIC Based on the Databricks example
# MAGIC [LangGraph MCP tool-calling agent](https://docs.databricks.com/aws/en/notebooks/source/generative-ai/langgraph-mcp-tool-calling-agent.html),
# MAGIC with the Unity Catalog registration and deployment steps intentionally omitted —
# MAGIC this notebook stops after authoring, testing, evaluating, and logging the agent.
# MAGIC
# MAGIC In this notebook, you:
# MAGIC - Author a LangGraph agent wrapped as an MLflow [`ResponsesAgent`](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html#mlflow.pyfunc.ResponsesAgent)
# MAGIC - Give it a **local Python executor tool** (a LangChain `@tool` that runs code
# MAGIC   in-process — replaces the managed-MCP `system.ai.python_exec` UC tool)
# MAGIC - Test the agent (single response + streaming) with MLflow tracing
# MAGIC - Evaluate it with MLflow GenAI scorers and log it as an MLflow model

# COMMAND ----------

# MAGIC %pip install -qqqq --force-reinstall databricks-sdk==0.105.0 mlflow==3.11.1 databricks-agents==1.9.4 databricks-langchain==0.19.0 langgraph==1.1.8 uv

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the agent code
# MAGIC
# MAGIC The agent is written to a local `agent.py` with `%%writefile` so it can be logged
# MAGIC as code. What it does:
# MAGIC
# MAGIC 1. **LLM**: `ChatDatabricks` pointing at the custom vLLM endpoint (the
# MAGIC    `AGENT_LLM_ENDPOINT` env var, set from the `ENDPOINT_NAME` widget before import).
# MAGIC    Tool binding works because the endpoint runs with `--enable-auto-tool-choice
# MAGIC    --tool-call-parser gemma4` (structured `tool_calls` in responses).
# MAGIC 2. **Tools**: a local `execute_python` LangChain `@tool` that runs code in-process
# MAGIC    and returns its stdout (demo-grade, unsandboxed). The original example's managed
# MAGIC    MCP setup (`system.ai.python_exec` via `DatabricksMultiServerMCPClient`) can be
# MAGIC    swapped back in when the UC tool is available.
# MAGIC 3. **Workflow**: a LangGraph `StateGraph` loops agent → tools → agent until the model
# MAGIC    stops requesting tool calls.
# MAGIC 4. **Wrapper**: `LangGraphResponsesAgent(ResponsesAgent)` adapts the graph to the
# MAGIC    Databricks Responses API with real streaming, and `mlflow.langchain.autolog()`
# MAGIC    traces every LLM call and tool invocation.

# COMMAND ----------

# MAGIC %%writefile agent.py
# MAGIC
# MAGIC import asyncio
# MAGIC import json
# MAGIC from typing import Annotated, Any, AsyncGenerator, Generator, Optional, Sequence, TypedDict, Union
# MAGIC
# MAGIC import mlflow
# MAGIC import nest_asyncio
# MAGIC from databricks_langchain import ChatDatabricks
# MAGIC from langchain.messages import AIMessage, AIMessageChunk, AnyMessage
# MAGIC from langchain_core.language_models import LanguageModelLike
# MAGIC from langchain_core.messages.tool import ToolMessage
# MAGIC from langchain_core.runnables import RunnableConfig, RunnableLambda
# MAGIC from langchain_core.tools import BaseTool, tool
# MAGIC from langgraph.graph import END, StateGraph
# MAGIC from langgraph.graph.message import add_messages
# MAGIC from langgraph.prebuilt.tool_node import ToolNode
# MAGIC from mlflow.pyfunc import ResponsesAgent
# MAGIC from mlflow.types.responses import (
# MAGIC     ResponsesAgentRequest,
# MAGIC     ResponsesAgentResponse,
# MAGIC     ResponsesAgentStreamEvent,
# MAGIC     output_to_responses_items_stream,
# MAGIC     to_chat_completions_input,
# MAGIC )
# MAGIC
# MAGIC nest_asyncio.apply()
# MAGIC
# MAGIC ############################################
# MAGIC ## LLM endpoint and system prompt
# MAGIC ############################################
# MAGIC # A custom endpoint deployed by this project, serving vLLM with native OpenAI tool
# MAGIC # calling (--tool-call-parser gemma4) so bind_tools() works. The notebook sets
# MAGIC # AGENT_LLM_ENDPOINT from its ENDPOINT_NAME widget before importing this module;
# MAGIC # the default applies when the logged model runs outside the notebook.
# MAGIC import os
# MAGIC
# MAGIC LLM_ENDPOINT_NAME = os.environ.get("AGENT_LLM_ENDPOINT", "gemma-4-e4b-it")
# MAGIC llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
# MAGIC
# MAGIC system_prompt = """
# MAGIC You are a helpful assistant that can run Python code. When a question involves
# MAGIC computation, use the Python code execution tool and base your answer on its result.
# MAGIC """
# MAGIC
# MAGIC ############################################
# MAGIC ## Tools — local Python executor
# MAGIC ############################################
# MAGIC # A local in-process Python interpreter tool (replaces the managed MCP
# MAGIC # system.ai.python_exec UC tool). NOTE: exec() runs unsandboxed in the agent's own
# MAGIC # process — fine for a demo agent; for production prefer a sandboxed executor such
# MAGIC # as the UC python_exec function.
# MAGIC @tool
# MAGIC def execute_python(code: str) -> str:
# MAGIC     """Execute Python code and return its printed output.
# MAGIC
# MAGIC     The code runs in a fresh namespace. Use print() to emit the values you need;
# MAGIC     only stdout is returned.
# MAGIC     """
# MAGIC     import contextlib, io, traceback
# MAGIC
# MAGIC     buf = io.StringIO()
# MAGIC     try:
# MAGIC         with contextlib.redirect_stdout(buf):
# MAGIC             exec(code, {}, {})
# MAGIC     except Exception:
# MAGIC         return f"Error executing code:\n{traceback.format_exc(limit=2)}"
# MAGIC     return buf.getvalue() or "(code ran successfully but printed no output)"
# MAGIC
# MAGIC
# MAGIC AGENT_TOOLS = [execute_python]
# MAGIC
# MAGIC
# MAGIC # The state for the agent workflow: the conversation plus any custom data.
# MAGIC class AgentState(TypedDict):
# MAGIC     messages: Annotated[Sequence[AnyMessage], add_messages]
# MAGIC     custom_inputs: Optional[dict[str, Any]]
# MAGIC     custom_outputs: Optional[dict[str, Any]]
# MAGIC
# MAGIC
# MAGIC def create_tool_calling_agent(
# MAGIC     model: LanguageModelLike,
# MAGIC     tools: Union[ToolNode, Sequence[BaseTool]],
# MAGIC     system_prompt: Optional[str] = None,
# MAGIC ):
# MAGIC     model = model.bind_tools(tools)
# MAGIC
# MAGIC     # Continue to the tools node while the model keeps requesting tool calls.
# MAGIC     def should_continue(state: AgentState):
# MAGIC         last_message = state["messages"][-1]
# MAGIC         if isinstance(last_message, AIMessage) and last_message.tool_calls:
# MAGIC             return "continue"
# MAGIC         return "end"
# MAGIC
# MAGIC     if system_prompt:
# MAGIC         preprocessor = RunnableLambda(
# MAGIC             lambda state: [{"role": "system", "content": system_prompt}] + state["messages"]
# MAGIC         )
# MAGIC     else:
# MAGIC         preprocessor = RunnableLambda(lambda state: state["messages"])
# MAGIC
# MAGIC     model_runnable = preprocessor | model
# MAGIC
# MAGIC     def call_model(state: AgentState, config: RunnableConfig):
# MAGIC         response = model_runnable.invoke(state, config)
# MAGIC         return {"messages": [response]}
# MAGIC
# MAGIC     workflow = StateGraph(AgentState)
# MAGIC     workflow.add_node("agent", RunnableLambda(call_model))
# MAGIC     workflow.add_node("tools", ToolNode(tools))
# MAGIC     workflow.set_entry_point("agent")
# MAGIC     workflow.add_conditional_edges(
# MAGIC         "agent",
# MAGIC         should_continue,
# MAGIC         {"continue": "tools", "end": END},
# MAGIC     )
# MAGIC     workflow.add_edge("tools", "agent")
# MAGIC     return workflow.compile()
# MAGIC
# MAGIC
# MAGIC # Wrap the compiled graph for compatibility with the Databricks Responses API.
# MAGIC class LangGraphResponsesAgent(ResponsesAgent):
# MAGIC     def __init__(self, agent):
# MAGIC         self.agent = agent
# MAGIC
# MAGIC     def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
# MAGIC         outputs = [
# MAGIC             event.item
# MAGIC             for event in self.predict_stream(request)
# MAGIC             if event.type == "response.output_item.done" or event.type == "error"
# MAGIC         ]
# MAGIC         return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)
# MAGIC
# MAGIC     async def _predict_stream_async(
# MAGIC         self,
# MAGIC         request: ResponsesAgentRequest,
# MAGIC     ) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
# MAGIC         cc_msgs = to_chat_completions_input([i.model_dump() for i in request.input])
# MAGIC         async for event in self.agent.astream(
# MAGIC             {"messages": cc_msgs}, stream_mode=["updates", "messages"]
# MAGIC         ):
# MAGIC             if event[0] == "updates":
# MAGIC                 for node_data in event[1].values():
# MAGIC                     if len(node_data.get("messages", [])) > 0:
# MAGIC                         all_messages = []
# MAGIC                         for msg in node_data["messages"]:
# MAGIC                             if isinstance(msg, ToolMessage) and not isinstance(msg.content, str):
# MAGIC                                 msg.content = json.dumps(msg.content)
# MAGIC                             all_messages.append(msg)
# MAGIC                         for item in output_to_responses_items_stream(all_messages):
# MAGIC                             yield item
# MAGIC             elif event[0] == "messages":
# MAGIC                 try:
# MAGIC                     chunk = event[1][0]
# MAGIC                     if isinstance(chunk, AIMessageChunk) and (content := chunk.content):
# MAGIC                         yield ResponsesAgentStreamEvent(
# MAGIC                             **self.create_text_delta(delta=content, item_id=chunk.id),
# MAGIC                         )
# MAGIC                 except Exception:
# MAGIC                     pass
# MAGIC
# MAGIC     def predict_stream(
# MAGIC         self, request: ResponsesAgentRequest
# MAGIC     ) -> Generator[ResponsesAgentStreamEvent, None, None]:
# MAGIC         agen = self._predict_stream_async(request)
# MAGIC         try:
# MAGIC             loop = asyncio.get_event_loop()
# MAGIC         except RuntimeError:
# MAGIC             loop = asyncio.new_event_loop()
# MAGIC             asyncio.set_event_loop(loop)
# MAGIC         ait = agen.__aiter__()
# MAGIC         while True:
# MAGIC             try:
# MAGIC                 item = loop.run_until_complete(ait.__anext__())
# MAGIC             except StopAsyncIteration:
# MAGIC                 break
# MAGIC             else:
# MAGIC                 yield item
# MAGIC
# MAGIC
# MAGIC def initialize_agent():
# MAGIC     """Initialize the agent with the local Python executor tool."""
# MAGIC     agent = create_tool_calling_agent(llm, AGENT_TOOLS, system_prompt)
# MAGIC     return LangGraphResponsesAgent(agent)
# MAGIC
# MAGIC
# MAGIC mlflow.langchain.autolog()
# MAGIC AGENT = initialize_agent()
# MAGIC mlflow.models.set_model(AGENT)

# COMMAND ----------

# Select the endpoint the agent talks to (job task passes ${var.endpoint_name}).
import os

dbutils.widgets.text("ENDPOINT_NAME", "gemma-4-e4b-it")
os.environ["AGENT_LLM_ENDPOINT"] = dbutils.widgets.get("ENDPOINT_NAME")

from agent import AGENT, LLM_ENDPOINT_NAME

print("Agent LLM endpoint:", LLM_ENDPOINT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the agent with MLflow GenAI scorers
# MAGIC
# MAGIC The evaluation exercises the agent directly (tool-requiring computations plus a
# MAGIC no-tool control question). `mlflow.langchain.autolog()` records a trace for every
# MAGIC step — open the trace UI from the cell output to inspect the tool calls made by the
# MAGIC Gemma endpoint. Extend `eval_dataset` as you iterate; metrics are tracked in MLflow.

# COMMAND ----------

import mlflow
from mlflow.genai.scorers import RelevanceToQuery, Safety

# Job runs have no notebook experiment, so traces (which mlflow.genai.evaluate relies
# on) have no destination and come back None, crashing the eval harness. Point MLflow
# at an explicit workspace experiment so tracing works both interactively and in jobs.
_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mlflow.set_experiment(f"/Users/{_user}/agent_implementation_example")

# Expectations must be nested under an "expectations" key (a top-level
# "expected_response" is a legacy format that crashes the eval harness in jobs):
# https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/eval-examples#evaluate-using-a-pandas-dataframe
eval_dataset = [
    {
        "inputs": {
            "input": [{"role": "user", "content": "Calculate the 15th Fibonacci number"}]
        },
        "expectations": {"expected_response": "The 15th Fibonacci number is 610."},
    },
    {
        "inputs": {
            "input": [{"role": "user", "content": "What is 847 plus 2956? Use Python."}]
        },
        "expectations": {"expected_response": "847 plus 2956 is 3803."},
    },
    {
        "inputs": {
            "input": [{"role": "user", "content": "Compute the sum of the first 100 prime numbers."}]
        },
        "expectations": {"expected_response": "The sum of the first 100 prime numbers is 24133."},
    },
    {
        "inputs": {
            "input": [{"role": "user", "content": "What is 20 factorial?"}]
        },
        "expectations": {"expected_response": "20 factorial is 2432902008176640000."},
    },
    {
        # No-tool control: the agent should answer directly without running code.
        "inputs": {
            "input": [{"role": "user", "content": "What is the capital of France?"}]
        },
        "expectations": {"expected_response": "The capital of France is Paris."},
    },
]

# The eval harness requires predict_fn to produce an MLflow trace per eval item
# (eval_item.trace is None otherwise, crashing expectation attachment in jobs).
# @mlflow.trace guarantees one regardless of autolog behavior in the job context.
@mlflow.trace
def traced_predict(input):
    return AGENT.predict({"input": input})

# Traces are exported asynchronously by default; in job runs the eval harness can lose
# the race and read the trace back as None. Force synchronous trace logging.
import os

os.environ["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"] = "false"

with mlflow.start_run(run_name="agent_eval"):
    eval_results = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=traced_predict,
        scorers=[RelevanceToQuery(), Safety()],
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the agent as an MLflow model
# MAGIC
# MAGIC Logged as code from `agent.py`, declaring the serving endpoint and UC function as
# MAGIC resources (enables automatic auth passthrough if the agent is ever deployed).
# MAGIC
# MAGIC **Unity Catalog registration and deployment are intentionally omitted** — see the
# MAGIC [original example](https://docs.databricks.com/aws/en/notebooks/source/generative-ai/langgraph-mcp-tool-calling-agent.html)
# MAGIC for those steps (`mlflow.register_model` + `databricks.agents.deploy`).

# COMMAND ----------

import mlflow
from agent import LLM_ENDPOINT_NAME
from mlflow.models.resources import DatabricksServingEndpoint
from pkg_resources import get_distribution

resources = [
    DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME),
]

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="agent.py",
        resources=resources,
        pip_requirements=[
            f"langgraph=={get_distribution('langgraph').version}",
            f"databricks-langchain=={get_distribution('databricks-langchain').version}",
        ],
    )

print("model_uri:", logged_agent_info.model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pre-deployment validation
# MAGIC
# MAGIC Rebuilds the model's environment with `uv` and runs a prediction against the logged
# MAGIC artifact — the same pre-deployment check the original example performs (slow, ~min).

# COMMAND ----------

mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/agent",
    input_data={"input": [{"role": "user", "content": "What is 7*6 in Python?"}]},
    env_manager="uv",
)
