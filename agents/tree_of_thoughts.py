from pydantic import BaseModel, Field
import operator
from collections import deque
from typing import Annotated, TypedDict, Optional, Literal
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import asyncio

load_dotenv()

class Plan(BaseModel):
    """Các bước để thực hiện một tác vụ."""

    steps: list[str] = Field(
        description="Các bước để thực hiện một tác vụ, mỗi bước là một chuỗi mô tả ngắn gọn.",
    )

system_prompt_template = (
    "Đưa ra các bước với tác vụ nhận được.\n"
    "Các bước là riêng lẻ, nếu đúng sẽ đưa ra câu trả lời đúng. "
    "Không thêm bất kỳ bước thừa nào.\n"
    "Kết quả của bước cuối cùng là câu trả lời cuối cùng, "
    "đảm bảo mỗi bước đều có thông tin đầy đủ, không bỏ qua bước nào."
)
planner_prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt_template),
     ("user", "Chuẩn bị các bước để thực hiện tác vụ:\n{task}\n")])

planner = planner_prompt | ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", temperature=1.0
).with_structured_output(Plan)

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

tools = load_tools(
    tool_names=["ddg-search", "arxiv", "wikipedia"],
    llm=llm
)

system_prompt = (
    "Bạn là một trợ lý thông minh, cẩn thận hỗ trợ giải quyết các nhiệm vụ phức tạp.\n"
    " Với một kế hoạch tổng thể để giải quyết nhiệm vụ và một bước cụ thể được giao, hãy tập trung thực hiện bước đó. "
    " Đừng giả định bất cứ điều gì — hãy luôn ghi nhớ rằng mọi thứ có thể thay đổi, và luôn cố gắng "
    "sử dụng các công cụ để kiểm tra lại thông tin.\n"
    "Hãy sử dụng công cụ Tìm kiếm (Search) để thu thập thông tin về các sự kiện thực tế, tin tức mới; "
    "dùng Arxiv để tham khảo các nghiên cứu gần đây và dùng Wikipedia để tra cứu kiến thức phổ thông."
)

step_template = (
    "Dựa trên nhiệm vụ và kế hoạch đã có, hãy thực hiện bước cụ thể dưới đây.\n"
    "NHIỆM VỤ:\n{task}\n\nKẾ HOẠCH:\n{previous_steps}\n\nBƯỚC CẦN THỰC HIỆN:\n{step}\n"
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", step_template),
])

execution_agent = prompt_template | create_react_agent(model=llm, tools=tools)

class TreeNode:

  def __init__(
        self,
        node_id: int,
        step: str,
        step_output: Optional[str] = None,
        parent: Optional["TreeNode"] = None,
    ):
        self.node_id = node_id
        self.step = step
        self.step_output = step_output
        self.parent = parent
        self.children = []
        self.final_response = None

  def __repr__(self):
    parent_id = self.parent.node_id if self.parent else "None"
    return f"Node_id: {self.node_id}, parent: {parent_id}, {len(self.children)} children."

  def get_full_plan(self) -> str:
    """Returns formatted plan with step numbers and past results."""
    steps = []
    node = self
    while node.parent:
      steps.append((node.step, node.step_output))
      node = node.parent

    full_plan = []
    for i, (step, result) in enumerate(steps[::-1]):
      if result:
        full_plan.append(f"# {i+1}. Planned step: {step}\nResult: {result}\n")
    return "\n".join(full_plan)
  
class PlanEvaluation(BaseModel):
  """Đánh giá kế hoạch"""

  is_final: bool = Field(
      description="Nhiệm vụ đã được giải quyết hay chưa. Kế hoạch chưa phải là cuối cùng nếu cần thêm bước.",
      default=False)

  score: float = Field(
      description="Điểm số từ 0.0 đến 1.0 đánh giá chất lượng kế hoạch và khả năng giải quyết nhiệm vụ bằng kế hoạch này.",
      default=False
  )


class PlanState(TypedDict):
    task: str
    root: TreeNode
    queue: deque[TreeNode]
    current_node: TreeNode
    next_node: TreeNode
    is_current_node_final: bool
    paths_explored: Annotated[int, operator.add]
    visited_ids: set[int]
    max_id: int
    candidates: Annotated[list[str], operator.add]
    best_candidate: str

class ReplanStep(BaseModel):
    """Bước kế tiếp được lập lại trong kế hoạch."""

    steps: list[str] = Field(
        description="các lựa chọn khác nhau cho bước kế tiếp được đề xuất"
    )

llm_replanner = llm.with_structured_output(ReplanStep)

replanner_prompt_template = (
    "Đề xuất hành động tiếp theo trong kế hoạch. Đừng thêm bất kỳ bước thừa nào.\n"
    "Nếu bạn nghĩ không cần thêm hành động nào, hãy trả về danh sách bước trống.\n"
    "NHIỆM VỤ: {task}\nCÁC BƯỚC TRƯỚC ĐÃ CÓ KẾT QUẢ: {current_plan}"

)
replanner_prompt = ChatPromptTemplate.from_messages(
    [("system", "Bạn là trợ lý thân thiện. Mục tiêu của bạn là giúp lập kế hoạch các hành động để giải quyết nhiệm vụ. Đừng tự giải quyết nhiệm vụ."),
     ("user", replanner_prompt_template)
    ]
)

replanner = replanner_prompt | llm_replanner

prompt_voting = PromptTemplate.from_template(
    "Chọn giải pháp tốt nhất cho một nhiệm vụ nhất định. "
    "\nNHIỆM VỤ: {task}\n\nCÁC GIẢI PHÁP:\n{candidates}\n"
)

def _vote_for_the_best_option(state):
  candidates = state.get("candidates", [])
  if not candidates:
    return {"best_response": None}
  all_candidates = []
  for i, candidate in enumerate(candidates):
    all_candidates.append(f"OPTION {i+1}: {candidate}")

  response_schema = {
      "type": "STRING",
      "enum": [str(i+1) for i in range(len(all_candidates))]}
  llm_enum = ChatGoogleGenerativeAI(
      model="gemini-2.5-flash", response_mime_type="text/x.enum",
      response_schema=response_schema)
  result = (prompt_voting | llm_enum | StrOutputParser()).invoke(
      {"candidates": "\n".join(all_candidates), "task": state["task"]}
  )
  return {"best_candidate": candidates[int(result)-1]}

final_prompt = PromptTemplate.from_template(
    "Bạn là một trợ lý hữu ích đã thực hiện một kế hoạch. "
    "Dựa trên kết quả của việc thực hiện, hãy chuẩn bị câu trả lời cuối cùng.\n"
    "Đừng suy đoán gì thêm.\nNHIỆM VỤ:\n{task}\n\nKẾ HOẠCH VÀ KẾT QUẢ:\n{plan}\n"
    "CÂU TRẢ LỜI CUỐI CÙNG:\n"
)

responder = final_prompt | llm | StrOutputParser()

async def _build_initial_plan(state: PlanState) -> PlanState:
  plan = await planner.ainvoke({"task": state["task"]})
  queue = deque()
  root = TreeNode(step=plan.steps[0], node_id=1)
  queue.append(root)
  current_root = root
  for i, step in enumerate(plan.steps[1:]):
    child = TreeNode(node_id=i+2, step=step, parent=current_root)
    current_root.children.append(child)
    queue.append(child)
    current_root = child
  return {"root": root, "queue": queue, "max_id": i+2}

async def _run_node(state: PlanState, config: RunnableConfig):
  node = state.get("next_node")
  visited_ids = state.get("visited_ids", set())
  queue = state["queue"]
  if node is None:
    while queue and not node:
      node = state["queue"].popleft()
      if node.node_id in visited_ids:
        node = None
    if not node:
      return Command(goto="vote", update={})

  step = await execution_agent.ainvoke({
      "previous_steps": node.get_full_plan(),
      "step": node.step,
      "task": state["task"]})
  node.step_output = step["messages"][-1].content
  visited_ids.add(node.node_id)
  return {"current_node": node, "queue": queue, "visited_ids": visited_ids, "next_node": None}

async def _plan_next(state: PlanState, config: RunnableConfig) -> PlanState:
  max_candidates = config["configurable"].get("max_candidates", 1)
  node = state["current_node"]
  next_step = await replanner.ainvoke({"task": state["task"], "current_plan": node.get_full_plan()})
  if not next_step.steps:
    return {"is_current_node_final": True}
  max_id = state["max_id"]
  for step in next_step.steps[:max_candidates]:
    child = TreeNode(node_id=max_id+1, step=step, parent=node)
    max_id += 1
    node.children.append(child)
    state["queue"].append(child)
  return {"is_current_node_final": False, "next_node": child, "max_id": max_id}

async def _get_final_response(state: PlanState) -> PlanState:
  node = state["current_node"]
  final_response = await responder.ainvoke({"task": state["task"], "plan": node.get_full_plan()})
  node.final_response = final_response
  return {"paths_explored": 1, "candidates": [final_response]}

def _should_create_final_response(state: PlanState) -> Literal["run", "generate_response"]:
  return "generate_response" if state["is_current_node_final"] else "run"

def _should_continue(state: PlanState, config: RunnableConfig) -> Literal["run", "vote"]:
  max_paths = config["configurable"].get("max_paths", 30)
  if state.get("paths_explored", 1) >= max_paths:
    return "vote"
  if state["queue"] or state.get("next_node"):
    return "run"
  return "vote"



builder = StateGraph(PlanState)
builder.add_node("initial_plan", _build_initial_plan)
builder.add_node("run", _run_node)
builder.add_node("plan_next", _plan_next)
builder.add_node("generate_response", _get_final_response)
builder.add_node("vote", _vote_for_the_best_option)

builder.add_edge(START, "initial_plan")
builder.add_edge("initial_plan", "run")
builder.add_edge("run", "plan_next")
builder.add_conditional_edges("plan_next", _should_create_final_response)
builder.add_conditional_edges("generate_response", _should_continue)
builder.add_edge("vote", END)

graph = builder.compile()

async def main():
    task = "Viết một trang kế hoạch về việc xây dựng một công ty khởi nghiệp AI"
    result = await graph.ainvoke({"task": task}, config={"recursion_limit": 10000, "configurable": {"max_paths": 3}})

    print(len(result["candidates"]))

    print(result["best_candidate"])

if __name__ == "__main__":
    asyncio.run(main())
