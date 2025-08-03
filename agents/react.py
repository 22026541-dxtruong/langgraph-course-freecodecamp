from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

tools = [add]

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash').bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=
        "You are my AI assistant, please answer my query to the best of your ability."
    )
    response = model.invoke([system_prompt] + list(state['messages']))
    return {'messages': [response]}

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    print("Last message:", last_message)
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        print("Tool call detected. Continue.")
        return 'continue'
    print("No tool call. Ending.")
    return 'end'

graph = StateGraph(AgentState)
graph.add_node('our_agent', model_call)
graph.set_entry_point('our_agent')

tool_node = ToolNode(tools=tools)
graph.add_node('tools', tool_node)

graph.add_conditional_edges(
    'our_agent',
    should_continue,
    {
        'continue': 'tools',
        'end': END
    }
)
graph.add_edge('tools', 'our_agent')

agent = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs: AgentState = {"messages": [HumanMessage(content="Add 40 + 12. Add 5 to the result.")]}
print_stream(agent.stream(inputs, stream_mode="values"))
