from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage
from config.config import settings
from core.state import State
from prompts.guardrail import GUARDRAIL_INSTRUCTIONS


class GuardrailsOutput(BaseModel):
    """Output model for the safety and relevance guardrail agent."""
    is_safe: bool = Field(description="True if the input is safe and relevant, False if it is a jailbreak attempt or irrelevant.")
    reasoning: str = Field(description="Brief explanation of the safety decision.")

guardrails_agent = ChatOpenAI(
    model=settings.llm.profiles["openai-guardrail"].model,
    temperature=settings.llm.profiles["openai-guardrail"].temperature,
    api_key=settings.llm.profiles["openai-guardrail"].api_key, 
).with_structured_output(GuardrailsOutput)

def guardrail(state: State, config: RunnableConfig):
    GUARDRAIL_ENABLED = settings.guardrail.enabled
    if not GUARDRAIL_ENABLED:
        return {
            "messages": [], 
            "summary": state.get("summary", None),
            "end_status": "continue"
        }

    user_input = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    
    prompt = f"{GUARDRAIL_INSTRUCTIONS}\n\nUser Input: {user_input[-1].content if user_input else ''}"
    guardrail_result = guardrails_agent.invoke(prompt)
    if not guardrail_result.is_safe:
        error_message = (
            f"🚨 I cannot assist with that request. Reason: {guardrail_result.reasoning}"
        )
        return {
            "messages": [AIMessage(content=error_message)],
            "end_status": "end" 
        }

    return {"messages": [], "end": False}
