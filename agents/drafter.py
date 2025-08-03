from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage, ToolMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()

document_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def update(content: str) -> str:
    """Update the document with content.
        Args:
            content (str): The content to update the document with.
    """
    global document_content
    document_content = content
    return f"Document updated: {content}"

@tool
def save(filenam: str) -> str:
    """Save the document to a file.
        Args:
            filenam (str): The name of the file to save the document.
    """
    global document_content
    if not filenam.endswith('.txt'):
        filenam += '.txt'
    try:
        with open(filenam, 'w') as file:
            file.write(document_content)
        return f"Document saved to {filenam}"
    except Exception as e:
        return f"Error saving document: {str(e)}"

tools = [update, save]

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash').bind_tools(tools)

def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
    """)
    if not state['messages']:
        user_input = "I'm ready to help you with your document. What would you like to do?"
        print(f"User: {user_input}")
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("\nWhat would you like to do?")
        print(f"User: {user_input}")
        user_message = HumanMessage(content=user_input)
    
    all_messages = [system_prompt] + list(state['messages']) + [user_message]
    response = model.invoke(all_messages)
    print(f"AI: {response.content}")
    if isinstance(response, AIMessage) and response.tool_calls:
        print(f'🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}')
    return {'messages': [user_message,response]}

def should_continue(state: AgentState) -> str:
    messages = state['messages']
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and 'saved' in str(message.content).lower() and 'document' in str(message.content).lower():
            print(f'Tool Message: {message}')
            return 'end'
    return 'continue'

def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")

graph = StateGraph(AgentState)
graph.add_node('our_agent', our_agent)
graph.set_entry_point('our_agent')

tool_node = ToolNode(tools=tools)
graph.add_node('tools', tool_node)

graph.add_edge('our_agent', 'tools')
graph.add_conditional_edges(
    'tools',
    should_continue,
    {
        'continue': 'our_agent',
        'end': END
    }
)

app = graph.compile()

def run_document_agent():
    print("\n ===== DRAFTER =====")
    
    state: AgentState = {"messages": []}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n ===== DRAFTER FINISHED =====")

if __name__ == "__main__":
    run_document_agent()
