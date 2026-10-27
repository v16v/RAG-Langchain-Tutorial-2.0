import os
from typing import Annotated, Sequence, TypedDict, Literal
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages    
from dotenv import load_dotenv
load_dotenv()   

openai_api_key = os.getenv("OPENAI_API_KEY")
# ---------------------------------------------------------
# 1. State Definition
# ---------------------------------------------------------
class DiligenceState(TypedDict):
    # A shared message bus that records interactions between agents
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # Specialized state attributes for focused storage
    company_name: str
    financial_summary: str
    risk_summary: str
    investment_memo: str
    next_step: str

# ---------------------------------------------------------
# 2. Supervisor Routing Schema
# ---------------------------------------------------------
# The Supervisor MUST reply with a structured decision indicating who works next.
class SupervisorDecision(BaseModel):
    next_worker: Literal["financial_analyst", "risk_auditor", "memo_writer", "FINISH"] = Field(
        description="The next specialized agent to execute, or 'FINISH' if the memo is completed."
    )
    instruction: str = Field(
        description="Clear instructions for the chosen worker on what to calculate or evaluate."
    )

llm = ChatOpenAI(model="gpt-4o", temperature=0.0, openai_api_key=openai_api_key)
supervisor_router = llm.with_structured_output(SupervisorDecision)

# ---------------------------------------------------------
# 3. The Supervisor Node
# ---------------------------------------------------------
def supervisor_node(state: DiligenceState):
    system_prompt = """You are the Lead Investment Partner at a venture fund.
Your job is to orchestrate a team of three specialists to evaluate a target company:
1. financial_analyst: Calculates revenue growth, operating margins, and unit economics.
2. risk_auditor: Evaluates legal risks, competitor threats, and regulatory hurdles.
3. memo_writer: Synthesizes both the financial and risk findings into a concise investment memo.

Guidelines:
- If financial data is missing, route to financial_analyst.
- If risk evaluation is missing, route to risk_auditor.
- Once both financial and risk analyses exist, route to memo_writer.
- Once the memo_writer has created the investment memo, return FINISH.
"""

    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
    decision: SupervisorDecision = supervisor_router.invoke(messages)
    
    return {
        "next_step": decision.next_worker,
        "messages": [HumanMessage(content=f"Supervisor assigned {decision.next_worker}: {decision.instruction}", name="Supervisor")]
    }

# ---------------------------------------------------------
# 4. Specialized Worker 1: Financial Analyst Agent
# ---------------------------------------------------------
def financial_analyst_node(state: DiligenceState):
    company = state["company_name"]
    prompt = f"""You are an elite Senior Financial Analyst.
Analyze the target company: '{company}'.
Provide estimated unit economics, revenue trajectory, gross margin profiles, and cash-flow sustainability.
Keep your analysis grounded, bulleted, and strictly focused on balance-sheet health."""

    response = llm.invoke([SystemMessage(content=prompt)] + list(state["messages"]))
    
    return {
        "financial_summary": response.content,
        "messages": [HumanMessage(content=response.content, name="FinancialAnalyst")]
    }

# ---------------------------------------------------------
# 5. Specialized Worker 2: Risk & Governance Auditor Agent
# ---------------------------------------------------------
def risk_auditor_node(state: DiligenceState):
    company = state["company_name"]
    prompt = f"""You are a Compliance & Risk Auditor.
Analyze potential failure points, regulatory scrutiny, market moat erosion, and operational bottlenecks for: '{company}'.
Deliver an objective assessment highlighting high, medium, and low risks."""

    response = llm.invoke([SystemMessage(content=prompt)] + list(state["messages"]))
    
    return {
        "risk_summary": response.content,
        "messages": [HumanMessage(content=response.content, name="RiskAuditor")]
    }

# ---------------------------------------------------------
# 6. Specialized Worker 3: Synthesis & Memo Writer Agent
# ---------------------------------------------------------
def memo_writer_node(state: DiligenceState):
    prompt = f"""You are the Chief Investment Officer.
Synthesize the provided findings into an Executive Investment Decision Memo.

Company: {state['company_name']}
Financial Health Findings:
{state['financial_summary']}

Risk and Governance Assessment:
{state['risk_summary']}

Format:
1. EXECUTIVE VERDICT (Invest / Pass / Request Terms)
2. KEY CATALYSTS
3. CRITICAL VULNERABILITIES
4. SUGGESTED DEAL STRUCTURE"""

    response = llm.invoke([SystemMessage(content=prompt)])
    
    return {
        "investment_memo": response.content,
        "messages": [HumanMessage(content="Final Investment Memo has been drafted and compiled.", name="MemoWriter")]
    }

# ---------------------------------------------------------
# 7. Graph Assembly & Conditional Edges
# ---------------------------------------------------------
workflow = StateGraph(DiligenceState)

# Add all nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("financial_analyst", financial_analyst_node)
workflow.add_node("risk_auditor", risk_auditor_node)
workflow.add_node("memo_writer", memo_writer_node)

# Entry point starts at the Supervisor
workflow.add_edge(START, "supervisor")

# Conditional routing from Supervisor based on next_worker decision
workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next_step"],
    {
        "financial_analyst": "financial_analyst",
        "risk_auditor": "risk_auditor",
        "memo_writer": "memo_writer",
        "FINISH": END
    }
)

# Every worker agent reports back to the Supervisor to determine the next action
workflow.add_edge("financial_analyst", "supervisor")
workflow.add_edge("risk_auditor", "supervisor")
workflow.add_edge("memo_writer", "supervisor")

# Compile the swarm
investment_swarm = workflow.compile()

# ---------------------------------------------------------
# 8. Execution Example
# ---------------------------------------------------------
if __name__ == "__main__":
    initial_payload = {
        "messages": [HumanMessage(content="Evaluate our upcoming deal: 'Acme Robotics', building AI autonomous forklifts.")],
        "company_name": "Acme Robotics",
        "financial_summary": "",
        "risk_summary": "",
        "investment_memo": "",
        "next_step": ""
    }

    print("\n--- 🚀 RUNNING MULTI-AGENT INVESTMENT SWARM ---\n")
    for event in investment_swarm.stream(initial_payload, stream_mode="values"):
        last_message = event["messages"][-1]
        print(f"[{last_message.name or 'User'}]: {last_message.content[:120]}...\n")

    print("--- 📑 FINAL EXECUTIVE INVESTMENT MEMO ---")
    final_state = investment_swarm.invoke(initial_payload)
    print(final_state["investment_memo"])