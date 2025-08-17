from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import Send
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, START
import asyncio
import base64

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class AgentState(TypedDict):
    video_uri: str
    chunks: int
    interval_secs: int
    summaries: Annotated[list, add_messages]
    final_summary: str

class _ChunkState(TypedDict):
    video_uri: str
    start_offset: int
    interval_secs: int

# Hàm xử lý từng đoạn video
async def _summarize_video_chunk(state: _ChunkState):
    start = state["start_offset"] * state["interval_secs"]
    end = (state["start_offset"] + 1) * state["interval_secs"]
    
    human_part = {
        "type": "text",
        "text": f"Please summarize the video from second {start} to {end}. "
                "Only use this portion to summarize, not the whole video."
    }

    video_part = {
        "type": "media",
        "data": state["video_uri"],
        "mime_type": "video/mp4"
    }

    response = await llm.ainvoke([HumanMessage(content=[human_part, video_part])])
    return {"summaries": [response.content]}

# Hàm chia đoạn cho video
def _map_summaries(state: AgentState):
    return [
        Send("summarize_video_chunk", {
            "video_uri": state["video_uri"],
            "interval_secs": state["interval_secs"],
            "start_offset": i
        })
        for i in range(state["chunks"])
    ]

# Gộp tất cả tóm tắt nhỏ thành 1 prompt lớn
def _merge_summaries(summaries: list[str], interval_secs: int = 600, **kwargs) -> str:
    sub_summaries = []
    for i, summary in enumerate(summaries):
        sub_summary = (
            f"Summary from sec {i * interval_secs} to sec {(i + 1) * interval_secs}:\n"
            f"{summary}\n"
        )
        sub_summaries.append(sub_summary)
    return "".join(sub_summaries)

# Prompt tổng hợp cuối cùng
reduce_prompt = PromptTemplate.from_template(
    "You are given a list of summaries of a video split into sequential pieces.\n"
    "SUMMARIES:\n{summaries}\n"
    "Based on that, prepare a summary of the whole video."
)

# Node xử lý tóm tắt cuối cùng
async def _generate_final_summary(state: AgentState):
    summary = _merge_summaries(
        summaries=state["summaries"], interval_secs=state["interval_secs"]
    )
    final_summary = await (reduce_prompt | llm | StrOutputParser()).ainvoke(
        {"summaries": summary}
    )
    return {"final_summary": final_summary}

# Xây dựng graph
graph = StateGraph(AgentState)
graph.add_node("summarize_video_chunk", _summarize_video_chunk)
graph.add_node("generate_final_summary", _generate_final_summary)
graph.add_conditional_edges(START, _map_summaries, ["summarize_video_chunk"])
graph.add_edge("summarize_video_chunk", "generate_final_summary")
graph.add_edge("generate_final_summary", END)
app = graph.compile()

# Chạy app
with open("agents/SampleVideo_1280x720_30mb.mp4", "rb") as f:
    video_base64 = base64.b64encode(f.read()).decode("utf-8")

async def main():
    result = await app.ainvoke(
        {"video_uri": video_base64, "chunks": 3, "interval_secs": 10},
        {"recursion_limit": 10}
    )
    print(result["final_summary"])

if __name__ == "__main__":
    asyncio.run(main())

# The video opens with a fluffy white rabbit emerging from its burrow into a lush, green field dotted with flowers. It 
# stretches, yawns, and happily explores its surroundings, sniffing flowers. A pink butterfly flutters by and lands gently 
# on the rabbit's nose, making it smile before flying off.

# An apple then falls from a nearby tree, which the rabbit picks up and takes a bite from. Unbeknownst to the rabbit, a 
# small brown squirrel, a larger red squirrel, and a chinchilla are observing it from the tree. The squirrels soon 
# begin to pelt the rabbit with various nuts and fruits, including apples and acorns, hitting its head and causing it 
# annoyance and confusion.

# As the rabbit continues to be bombarded, the butterfly lands on its head. The squirrels' attacks escalate, becoming more
# targeted towards the butterfly. The red squirrel eventually grabs the butterfly, rips off one of its wings, and throws the 
# detached wing. A spiky green fruit then appears, which the squirrels also throw at the rabbit, causing it pain and distress. 
# The video concludes with the squirrels appearing pleased with their pranks, while the rabbit is left clutching its chest in 
# discomfort, surrounded by fallen fruits and nuts.
