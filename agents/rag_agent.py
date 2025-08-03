from typing import Annotated, Sequence, TypedDict
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, HumanMessage
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0.1)

embeddings = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')

pdf_path = 'Stock_Market_Performance_2024.pdf'

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found at {pdf_path}. Please provide a valid path.")

pdf_loader = PyPDFLoader(pdf_path)

try:
    pages = pdf_loader.load()
    print(f"Loaded {len(pages)} pages from the PDF.")
except Exception as e:
    print(f"Failed to load PDF: {e}")
    raise

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

pages_split = text_splitter.split_documents(pages)

persist_directory = r"."
collection_name = "stock_market"

try:
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print(f"Vectorstore created with {len(pages_split)} chunks.")
except Exception as e:
    print(f"Failed to create vectorstore: {e}")
    raise

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

@tool
def retriever_tool(query: str) -> str:
    """This tool searches and returns the information from the Stock Market Performance 2024 document."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found in the document."
    
    result = []
    for i, doc in enumerate(docs):
        result.append(f"Document {i+1}:\n{doc.page_content}")
    return "\n\n".join(result)

tools = [retriever_tool]

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState) -> bool:
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

tools_dict = {our_tool.name: our_tool for our_tool in tools}

def call_llm(state: AgentState) -> AgentState:
    messages = [SystemMessage(content=system_prompt)] + list(state['messages'])
    messages = llm.invoke(messages)
    return {'messages': [messages]}

def take_acion(state: AgentState) -> AgentState:
    tool_calls = state['messages'][-1].tool_calls
    results =[]
    for tool_call in tool_calls:
        print(f"Calling tool: {tool_call['name']} with args: {tool_call['args'].get('query', '')}")
        if not tool_call['name'] in tools_dict:
            print(f"Tool {tool_call['name']} not found.")
            result = "Tool not found."
        else:
            tool = tools_dict[tool_call['name']]
            result = tool.invoke(tool_call['args'].get('query', ''))
            print(f"Tool result: {result}")
        results.append(ToolMessage(
            content=result,
            name=tool_call['name'],
            tool_call_id=tool_call['id']
        ))
    print('Tools Execution Complete. Back to the model.')
    return {'messages': results}

graph = StateGraph(AgentState)
graph.add_node("call_llm", call_llm)
graph.add_node("take_action", take_acion)

graph.set_entry_point('call_llm')
graph.add_conditional_edges(
    'call_llm',
    should_continue,
    {
        True: 'take_action',
        False: END
    }
)
graph.add_edge('take_action', 'call_llm')  # Loop back to call_llm after taking action

rag_agent = graph.compile()

def running_agent():
    print("\n=== RAG AGENT===")
    
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages = [HumanMessage(content=user_input)] # converts back to a HumanMessage type

        result = rag_agent.invoke({"messages": messages})
        
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)


running_agent()
