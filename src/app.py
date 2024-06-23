import logging
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import streamlit as st
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
  db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
  logger.info(f"Initializing database connection with URI: {db_uri}")
  return SQLDatabase.from_uri(db_uri)

def get_sql_chain(db):
  template = """
    You are a data analyst at a financial company. You are interacting with a user who is asking you questions about their account and portfolio.
    Based on the table schema below, write a MySQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the MySQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For example:
    Question: Tell me about my account holdings
    SQL Query: SELECT data from okx_accounts;
    Question: Show my portfolio performance for last month
    SQL Query: SELECT * FROM metrics WHERE created_at >= DATE_SUB(CURDATE(), INTERVAL 1 MONTH);
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """
    
  prompt = ChatPromptTemplate.from_template(template)
  
  llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
  
  def get_schema(_):
    logger.info("Fetching table schema.")
    return db.get_table_info()
  
  return (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | llm
    | StrOutputParser()
  )

def is_generic_query(query):
    generic_patterns = [
        r"how can you help", 
        r"what can you do", 
        r"help me", 
        r"what are your capabilities", 
        r"assist me",
        r"who are you"
    ]
    for pattern in generic_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return True
    return False
    
def get_response(user_query: str, db: SQLDatabase, chat_history: list):
  logger.info(f"Received user query: {user_query}")
  sql_chain = get_sql_chain(db)

  if is_generic_query(user_query):
        logger.info(f"Returning generic answer")
        return """I'm an AI assistant and I can help you with queries about your account and portfolio, 
        such as account holdings, portfolio performance, and more. Ask me anything related to your financial data."""
  
  template = """
    You are a data analyst at a financial company. You are interacting with a user who is asking you questions about their account and portfolio.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}"""
  
  prompt = ChatPromptTemplate.from_template(template)
  
  llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
  
  try:
    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    generated_query = sql_chain.invoke({"question": user_query, "chat_history": chat_history})
    logger.info(f"Generated SQL Query: {generated_query}")
    response = chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })
    logger.info(f"LLM Response: {response}")
  except Exception as e:
        logger.error(f"Error in generating response: {e}")
        response = f"I can't help you with this question at this time. I'm constantly evoloving and learning more, please try again later"
  return response

  
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
      AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database."),
    ]

load_dotenv()

st.set_page_config(page_title="Chat with QuantBot", page_icon=":speech_balloon:")

st.title("Chat with QuantBot")

with st.sidebar:
    st.subheader("Settings")
    st.write("Connect to the database and start chatting.")
    
    st.text_input("Host", value="localhost", key="Host")
    st.text_input("Port", value="3306", key="Port")
    st.text_input("User", value="root", key="User")
    st.text_input("Password", type="password", value="admin", key="Password")
    st.text_input("Database", value="Quantbot", key="Database")
    
    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            db = init_database(
                st.session_state["User"],
                st.session_state["Password"],
                st.session_state["Host"],
                st.session_state["Port"],
                st.session_state["Database"],
            )
            st.session_state.db = db
            st.success("Connected to database!")
    
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
        
    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
        st.markdown(response)
        
    st.session_state.chat_history.append(AIMessage(content=response))