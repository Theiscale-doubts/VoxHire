import dotenv
dotenv.load_dotenv()
from typing import Dict, List, Any

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


from langchain_google_genai import ChatGoogleGenerativeAI  # assuming this stays
from tools import save_qa_tool



# Initialize LLM model instance
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
)
session_domains = {}
# Session store for histories
session_store: Dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

SYSTEM_PROMPT = """
You are Synthia, an expert interviewer.

RULES:
1) First message = introduction → Greet briefly + ask first domain question
2) Ongoing → Evaluate their answer + ask ONE follow-up domain question

STRICT: Only ask domain questions. Never deviate from this domain.

DOMAIN NOTES:
- Data Analytics: Focus on SQL, data cleaning, visualization, business metrics. NO statistical theory/formulas.
- Data Science: Include ML, statistics, feature engineering.
- Software Engineering: Code, system design, debugging.
- Machine Learning: Algorithms, training, deployment.

Keep questions:
- Clear & specific
- Practical, not theoretical
- Progressive (basic → advanced)
- One at a time
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])


agent = prompt.partial(system_prompt=SYSTEM_PROMPT) | llm

# Wrap with memory/history
agent_with_memory = RunnableWithMessageHistory(
    agent,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

def run_agent_turn(message: str, session_id: str, domain: str | None = None):
    # Store domain for session
    if session_id not in session_domains and domain:
        session_domains[session_id] = domain

    # Inject domain into system prompt
    domain_text = session_domains.get(session_id, "general")
    print("Domain for session:", domain_text)
    system_prompt = SYSTEM_PROMPT.format(domain=domain) + f"\n\nYou are interviewing for the domain: {domain_text}. Keep all questions strictly within this domain."
    
    result = agent_with_memory.invoke(
        {
            "input": message,
            "system_prompt": system_prompt
        },
        config={"configurable": {"session_id": session_id}}
    )
    print("Agent result:", result)
    return result.content
