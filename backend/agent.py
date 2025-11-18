import dotenv
dotenv.load_dotenv()
from typing import Dict, List, Any
import random

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_google_genai import ChatGoogleGenerativeAI
from tools import save_qa_tool

# Initialize LLM model instance with HIGHER temperature for more variety
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.9,  # Increased from 0.7 for more variety
)

session_domains = {}
session_topics_covered = {}  # NEW: Track covered topics per session
session_question_count = {}  # NEW: Track question count

# Session store for histories
session_store: Dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

# Domain-specific question banks with topic tags
DOMAIN_QUESTIONS = {
    "datascience": {
        "topics": [
            "Statistics (distributions, hypothesis testing, p-values, confidence intervals, correlation vs causation)",
            "Machine Learning basics (supervised/unsupervised learning, model evaluation metrics)",
            "Feature engineering and selection techniques",
            "Data preprocessing (missing data, outliers, scaling/normalization)",
            "A/B testing and experimental design",
            "Model validation (cross-validation, overfitting/underfitting)",
            "Python concepts (pandas, numpy, scikit-learn - concepts only, NO code writing)",
            "Real-world ML scenarios (deployment, algorithm selection)"
        ],
        "sample_starters": [
            "Explain the difference between Type I and Type II errors",
            "How do you evaluate a classification model's performance?",
            "What techniques would you use for feature selection?",
            "How do you handle missing data in a dataset?",
            "Explain the concept of overfitting and how to prevent it",
            "What's the difference between correlation and causation?",
            "How would you design an A/B test for a new feature?"
        ]
    },
    "hr(humain recourse) + managerial": {
        "topics": [
            "Behavioral questions (challenging projects, difficult team members)",
            "Situational scenarios (unclear requirements, falling behind schedule)",
            "Soft skills and cultural fit",
            "Teamwork and communication",
            "Career goals and motivation",
            "Strengths and development areas",
            "Conflict resolution and feedback"
        ],
        "sample_starters": [
            "Tell me about a time you had to learn a new technology quickly",
            "How do you handle stress in a fast-paced environment?",
            "Describe a situation where you disagreed with your manager",
            "What motivates you in your career?",
            "How do you explain technical concepts to non-technical stakeholders?",
            "Tell me about a project that failed. What did you learn?",
            "Why do you want to work here?"
        ]
    },
    "data analytics": {
        "topics": [
            "SQL",
            "Data visualization (chart selection, dashboard design, storytelling)",
            "Business metrics (KPIs, funnel analysis)",
            "Data cleaning (missing values, duplicates, quality checks)",
            "Tools concepts (Excel, Tableau, Power BI)",
            "Exploratory Data Analysis techniques",
            "Python for data manipulation (pandas concepts - NO code writing)",
            "Reporting and stakeholder communication"
        ],
        "sample_starters": [
            "How do you choose the right visualization for different data types?",
            "Explain window functions in SQL and when you'd use them",
            "What KPIs would you track for an e-commerce website?",
            "How do you handle duplicate records in a dataset?",
            "Explain the difference between a dashboard and a report",
            "What's your approach to exploratory data analysis?",
            "How would you optimize a slow SQL query?"
        ]
    },
    "product": {
        "topics": [
            "Product strategy and roadmap planning",
            "User research and customer discovery",
            "Prioritization frameworks (RICE, MoSCoW, Kano)",
            "Feature definition and user stories",
            "Metrics and success measurement",
            "Stakeholder management",
            "A/B testing and experimentation",
            "Competitive analysis and market positioning"
        ],
        "sample_starters": [
            "How do you prioritize features on a product roadmap?",
            "Explain how you would conduct user research for a new feature",
            "What metrics would you use to measure product success?",
            "How do you handle conflicting stakeholder requirements?",
            "Describe the RICE prioritization framework",
            "How would you analyze a competitor's product?",
            "What's your approach to writing user stories?"
        ]
    },
    "frontend": {
        "topics": [
            "HTML/CSS (semantic HTML, flexbox, grid, responsive design)",
            "JavaScript fundamentals (ES6+, async/await, closures)",
            "React/Vue/Angular concepts (lifecycle, state, hooks)",
            "Performance optimization (lazy loading, code splitting)",
            "Web accessibility (a11y) best practices",
            "Browser APIs and DOM manipulation",
            "CSS approaches and preprocessors",
            "Testing concepts (unit, integration)"
        ],
        "sample_starters": [
            "Explain the difference between flexbox and CSS grid",
            "How do React hooks improve upon class components?",
            "What are the key principles of web accessibility?",
            "How would you optimize the performance of a slow webpage?",
            "Explain event bubbling and capturing",
            "What's the difference between var, let, and const?",
            "How do you ensure your website is responsive?"
        ]
    },
    "devops": {
        "topics": [
            "CI/CD pipelines and automation",
            "Containerization (Docker, orchestration basics)",
            "Cloud platforms (AWS/Azure/GCP services)",
            "Infrastructure as Code (Terraform, CloudFormation)",
            "Monitoring and logging (metrics, alerts, observability)",
            "Version control (Git workflows, branching)",
            "Linux/Unix and shell scripting concepts",
            "Security (secrets management, access control)",
            "Incident management and troubleshooting"
        ],
        "sample_starters": [
            "Explain the benefits of containerization",
            "How would you set up a CI/CD pipeline?",
            "What's the difference between monitoring and observability?",
            "Describe Infrastructure as Code and its benefits",
            "How do you manage secrets in a production environment?",
            "Explain different Git branching strategies",
            "How would you troubleshoot a production incident?"
        ]
    }
}

SYSTEM_PROMPT = """
You are Synthia, an expert interviewer conducting a {total_questions}-question interview.

CRITICAL VARIETY RULES:
- This is question {current_question} of {total_questions}
- You have already asked about: {covered_topics}
- You MUST ask about a NEW topic that is NOT in the covered list above
- NEVER repeat topics or ask similar questions to what's already been asked
- Each question should explore a DIFFERENT area from the domain

{domain_context}

TOPIC ROTATION MANDATE:
- Deliberately choose topics you HAVEN'T covered yet
- If you've asked about SQL JOINs, move to visualization or business metrics
- If you've asked about React hooks, move to CSS or performance
- Space out related topics - don't cluster similar subjects together

QUESTION STYLE:
- Keep questions SHORT and DIRECT (1-2 sentences maximum)
- Ask ONE specific thing at a time
- For Python: CONCEPTS ONLY - NO code writing requests
- Mix conceptual and practical questions
- Gradually increase difficulty as interview progresses

EXAMPLES OF GOOD VARIETY this is only for exampledont just folow the exact flow :
Question 1: SQL JOINs → Question 2: Data visualization → Question 3: Business metrics → Question 4: Python pandas concepts

LANGUAGE: Always communicate in English only.

RESPONSE FORMAT:
1. Brief evaluation of their previous answer (if applicable)
2. ONE new question from an UNCOVERED topic area
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

agent = prompt | llm

# Wrap with memory/history
agent_with_memory = RunnableWithMessageHistory(
    agent,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

def run_agent_turn(message: str, session_id: str, domain: str | None = None):
    # Initialize session tracking
    if session_id not in session_domains and domain:
        session_domains[session_id] = domain
        session_topics_covered[session_id] = []
        session_question_count[session_id] = 0

    # Get domain for this session
    domain_text = session_domains.get(session_id, "general")
    print("Domain for session:", domain_text)
    
    # Increment question count
    session_question_count[session_id] += 1
    current_q = session_question_count[session_id]
    
    # Get domain-specific context
    domain_info = DOMAIN_QUESTIONS.get(domain_text.lower(), {})
    topics_list = domain_info.get("topics", [])
    
    # Build domain context with available topics
    domain_context = f"AVAILABLE TOPICS FOR {domain_text.upper()}:\n"
    for i, topic in enumerate(topics_list, 1):
        domain_context += f"{i}. {topic}\n"
    
    # Add some starter questions for inspiration (randomized)
    sample_starters = domain_info.get("sample_starters", [])
    if sample_starters:
        random_samples = random.sample(sample_starters, min(3, len(sample_starters)))
        domain_context += f"\nEXAMPLE QUESTIONS (for inspiration, vary your wording):\n"
        for sample in random_samples:
            domain_context += f"- {sample}\n"
    
    # Track covered topics
    covered = session_topics_covered.get(session_id, [])
    covered_str = ", ".join(covered) if covered else "None yet"
    
    # Create system prompt with session context
    system_prompt = SYSTEM_PROMPT.format(
        domain_context=domain_context,
        covered_topics=covered_str,
        current_question=current_q,
        total_questions="8-10"
    )
    
    system_prompt += f"\n\nSTRICT DOMAIN: {domain_text}. Ask ONLY {domain_text} questions. Ignore off-topic responses."
    
    result = agent_with_memory.invoke(
        {
            "input": message,
            "system_prompt": system_prompt
        },
        config={"configurable": {"session_id": session_id}}
    )
    
    # Extract topic from result and add to covered topics (simple heuristic)
    # You might want to enhance this with more sophisticated topic extraction
    result_text = result.content.lower()
    for topic in topics_list:
        topic_keywords = topic.split('(')[0].strip().lower()
        if any(keyword in result_text for keyword in topic_keywords.split()):
            if topic not in covered:
                session_topics_covered[session_id].append(topic.split('(')[0].strip())
                break
    
    print(f"Question {current_q}: Covered topics so far: {session_topics_covered[session_id]}")
    
    return result.content
