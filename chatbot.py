import os
import asyncio
import wikipediaapi
from dotenv import load_dotenv

# LangChain & OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.output_parsers import StrOutputParser

# FAISS (RAG for document retrieval)
from langchain_community.document_loaders import PyPDFLoader  
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Tavily for real-time search
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import Tool
from langgraph.prebuilt import create_react_agent

# sqlite for memory storage to keep tabs on past conversations
import sqlite3

# Load API Keys from .env
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# Load OpenAI model
model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))

# Load and process documents (FAISS Vector Store)
pdf_loader = PyPDFLoader("testRAG.pdf")
text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=50)
docs = text_splitter.split_documents(pdf_loader.load())

# Create FAISS vector store
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(docs, embeddings)
vector_store.save_local("faiss_index")  # Optional: Save for persistence

# Define a transformer to summarize past conversation data
summarizer = StrOutputParser()

# Set up Tavily Search
tavily_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True
)

# Wrap Tavily as a LangChain tool
search_tool = Tool(
    name="Web Search",
    func=tavily_tool.invoke,
    description="Search the web for real-time information."
)

# Wikipedia Search (Only when explicitly mentioned)
def search_wikipedia(query):
    """Search Wikipedia only if 'Wikipedia' is mentioned in the query."""
    if "wikipedia" not in query.lower():
        return None  # Do nothing if Wikipedia isn't mentioned

    clean_query = query.lower().replace("wikipedia", "").strip()
    wiki = wikipediaapi.Wikipedia(language="en", user_agent="MyChatbot/1.0 (sofie.seilnacht@berkeley.edu)")
    page = wiki.page(clean_query)

    if page.exists():
        print("\n🌍 Wikipedia Search Found:\n", page.summary[:750])
        return page.summary[:750]  # Return first 750 characters
    else:
        print("\n⚠️ No Wikipedia article found for:", clean_query)
        return "Sorry, no Wikipedia article found on that topic."

# Create REACT Agent (Decides which tool to use)
agent = create_react_agent(
    model,
    tools=[search_tool]  # Add more tools like FAISS or Wikipedia if needed
)

# Define conversation memory per user
user_conversations = {}

# Define Trimmer (to prevent token overload)
trimmer = trim_messages(
    max_tokens=1024,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect("chat_memory.db")
cursor = conn.cursor()

# Create a table to store chat history
cursor.execute("""
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    message TEXT,
    response TEXT
)
""")
conn.commit()

def get_recent_past_conversations(user_id, limit=10):
    """Fetch the last 'limit' messages from the database."""
    
    # Debug: Print the query being executed
    print(f"Retrieving last {limit} messages for user: {user_id}")

    cursor.execute("""
        SELECT message, response FROM chat_history
        WHERE user_id = ? 
        ORDER BY id DESC 
        LIMIT ?
    """, (user_id, int(limit)))  # ✅ Ensure limit is an integer
    
    results = cursor.fetchall()
    
    # Debug: Print retrieved results
    print(f"Retrieved {len(results)} messages: {results}")
    
    return results



def save_to_db(user_id, message, response):
    """Stores user messages and chatbot responses in SQLite."""
    cursor.execute("""
        INSERT INTO chat_history (user_id, message, response)
        VALUES (?, ?, ?)
    """, (user_id, message, response))
    conn.commit()


async def call_model(state: MessagesState):
    """Handles chatbot response asynchronously using the REACT agent with real-time streaming."""

    # Trim messages to prevent token overload
    trimmed_messages = trimmer.invoke(state["messages"])
    
    # Extract latest user query
    latest_query = trimmed_messages[-1].content  

    # ✅ Handle Recall Requests & Summarize Only the Last 10 Messages
    if any(keyword in latest_query.lower() for keyword in ["recall", "remember", "look back"]):
        user_id = state.get("user_id", "default_user")  # Get user ID
        past_chats = get_recent_past_conversations(user_id, limit=10)  # ✅ Only fetch last 10 messages
        
        if past_chats:
            # ✅ Convert past messages into a format for summarization
            chat_history_text = "\n".join([f"User: {msg} | Bot: {resp}" for msg, resp in past_chats])

            # ✅ Summarize past interactions into a **single, clear response**
            summary = summarizer.parse(f"Summarize user past interactions and provide a direct answer:\n{chat_history_text}")
            
            # ✅ Modify query to reflect the recall summary, ensuring a **direct answer**
            query = f"Based on this summary: {summary}, {latest_query}"
        else:
            return {"messages": trimmed_messages + [AIMessage(content="I don't seem to have past records.")]}  # ✅ Early return if no records exist

    # ✅ Streaming Chatbot Response
    print("\n🤖 Chatbot:", end=" ", flush=True)
    response_text = ""

    async for chunk in model.astream(trimmed_messages + [HumanMessage(content=query)]):  # ✅ Append the modified query with recall summary
        chunk_content = chunk.get("messages", [{}])[-1].get("content", "") or chunk.get("text", "")

        if chunk_content:
            print(chunk_content, end="", flush=True)  # ✅ Stream response
            response_text += chunk_content  # ✅ Store full response

    print("\n")  # Newline after response finishes

    # ✅ Store final response in chat history & return
    response = AIMessage(content=response_text.strip())

    return {"messages": trimmed_messages + [response]}  # Append response to chat history

workflow = StateGraph(state_schema=MessagesState)
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

# Explicitly define required keys
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

async def chat():
    """Runs an interactive chatbot session in the terminal with user tracking."""
    
    # Ask for user ID
    user_id = input("\nEnter your user ID: ").strip()
    if user_id.lower() == "bye":
        print("\n👋 Chatbot session ended. Goodbye!\n")
        return

    # Ensure user has a conversation history
    if user_id not in user_conversations:
        user_conversations[user_id] = []

    print(f"\n💬 {user_id}, your chatbot session has started! Type 'bye' to stop.\n")

    while True:
        query = input(f"{user_id}: ").strip()
        if query.lower() == "bye":
            print(f"\n👋 Chatbot session ended for {user_id}. Goodbye!\n")
            break

        user_conversations[user_id].append(HumanMessage(content=query))

        # ✅ Handle Recall Requests & Summarize Past Conversations into One Answer
        if any(keyword in query.lower() for keyword in ["recall", "remember", "look back"]):
            past_chats = get_recent_past_conversations(user_id, limit=10)  # ✅ STRICTLY last 10 messages

            if past_chats:
                # ✅ Prepare chat history as context (DO NOT PRINT THIS)
                chat_history_text = "\n".join([f"User: {msg} | Bot: {resp}" for msg, resp in past_chats])

                # ✅ Summarize past convos but ONLY for context, NOT output
                summary = summarizer.parse(
                    f"""
                    Here is the past conversation for your own reference: 
                    {chat_history_text} 
                    
                    Based on this, provide a SINGLE, DIRECT response to the user's latest query without repeating history.
                    """
                )

                # ✅ Replace query with a clear direct answer
                response_text = summary.strip()
                print(f"\n🤖 Chatbot: {response_text}\n")
                continue  # ✅ Skip normal chatbot response handling
            else:
                print("\n🤖 Chatbot: I don't seem to have past records.\n")
                continue  # ✅ Skip response generation if no past records


        try:
            async for chunk in app.astream(
                {"messages": user_conversations[user_id] + [HumanMessage(content=query)]},
                {
                    "thread_id": user_id,  # Required
                    "checkpoint_ns": "chatbot",  # Static namespace for session tracking
                    "checkpoint_id": f"{user_id}_{len(user_conversations[user_id])}"  # Unique checkpoint per turn
                }
            ):
                chunk_content = chunk.get("messages", [{}])[-1].get("content", "") or chunk.get("text", "")

                if chunk_content:  # ✅ Stream response only if valid
                    print(chunk_content, end="", flush=True)
                    response_text += chunk_content  # ✅ Collect response

        except Exception as e:
            print(f"\n⚠️ Error during response streaming: {str(e)}")
            response_text = "Sorry, an error occurred while processing your request."

        print("\n")  # Newline after response finishes

        # ✅ Store final response
        chatbot_response = AIMessage(content=response_text.strip())
        user_conversations[user_id].append(chatbot_response)
        save_to_db(user_id, query, response_text.strip())  # ✅ Save clean data

# Run the chatbot asynchronously
asyncio.run(chat())
