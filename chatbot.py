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
    """Fetch the last 'limit' messages from the database, excluding recall requests."""
    
    # Debug: Print the query being executed
    print(f"Retrieving last {limit} messages for user: {user_id}")

    cursor.execute("""
        SELECT message, response FROM chat_history
        WHERE user_id = ? 
        ORDER BY id DESC
        LIMIT ?
    """, (user_id, int(limit)))  # ✅ Ensure limit is an integer
    
    results = cursor.fetchall()

    # Debug: Print raw retrieved results before filtering
    print(f"Retrieved {len(results)} messages (before filtering): {results}")

    # 🔹 Filter out messages where the user input contains "recall" (case-insensitive)
    filtered_results = [msg for msg in results if "recall" not in msg[0].lower()]

    # Get only the last `limit` messages **after filtering** to ensure correct count
    final_results = filtered_results[-limit:]

    # Debug: Print final retrieved results after filtering
    print(f"Final {len(final_results)} messages (after filtering): {final_results}")

    return final_results




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

    # ✅ Handle Recall Requests ONLY When Necessary
    if any(keyword in latest_query.lower() for keyword in ["recall", "remember", "look back"]):
        user_id = state.get("user_id", "default_user")  # Get user ID
        past_chats = get_recent_past_conversations(user_id, limit=10)  # ✅ Only fetch last 10 messages

        if past_chats:
            # ✅ Convert past messages into a formatted summary
            chat_history_text = "\n".join([f"User: {msg} | Bot: {resp}" for msg, resp in past_chats])
            summary = summarizer.parse(f"Summarize user past interactions and provide a direct answer:\n{chat_history_text}")
            
            # ✅ Modify query to reflect the recall summary, ensuring a **direct answer**
            query = f"Based on this summary: {summary}, {latest_query}"
        else:
            return {"messages": trimmed_messages + [AIMessage(content="I don't seem to have past records.")]}  # ✅ Early return if no records exist

    else:
        # ✅ If no recall is requested, process query as usual
        query = latest_query  # Keep the original input

    # ✅ Streaming Chatbot Response
    print("\n🤖 Chatbot:", end=" ", flush=True)
    response_text = ""

    async for chunk in model.astream(trimmed_messages + [HumanMessage(content=query)]):
        if hasattr(chunk, "content"):  # ✅ Check if 'chunk' has a 'content' attribute
            chunk_content = chunk.content  # ✅ Directly access 'content'
        else:
            chunk_content = ""

        if chunk_content:
            print(chunk_content, end="", flush=True)  # ✅ Stream response
            response_text += chunk_content  # ✅ Collect response text


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
        response_text = ""  # ✅ Always initialize before using

        # ✅ CHECK IF QUERY REQUIRES RECALL (ONLY THEN ACCESS DATABASE)
        if any(keyword in query.lower() for keyword in ["recall", "remember"]):
            try:
                past_chats = get_recent_past_conversations(user_id, limit=10)  

                if past_chats:
                    chat_history_text = "\n".join([f"User: {msg} | Bot: {resp}" for msg, resp in past_chats])

                    # ✅ Generate a **direct answer** based on past conversation
                    recall_prompt = f"""
                    The user wants to recall past conversations. 
                    Here are the last few exchanges:
                    
                    {chat_history_text}

                    User's latest question: "{query}"

                    🔹 Answer clearly and concisely, referencing past interactions. 
                    🔹 If the user's query is about their age, location, or facts about themselves, return an exact answer.
                    🔹 DO NOT summarize vaguely—give a factual answer.
                    """

                    response = model.invoke([HumanMessage(content=recall_prompt)])  # ✅ Force direct response
                    response_text = response.content.strip()

                    print(f"\n🔍 Recall Triggered. Using past memory for response.\n")
                    print(f"\n🤖 Chatbot: {response_text}\n")  # ✅ Print response immediately

                    chatbot_response = AIMessage(content=response_text)
                    user_conversations[user_id].append(chatbot_response)
                    save_to_db(user_id, query, chatbot_response.content)  
                    continue  # ✅ Skip normal chatbot flow after recall

                else:
                    print("\n🤖 Chatbot: I don't seem to have past records.\n")
                    continue  

            except Exception as e:
                print(f"\n⚠️ Error fetching past conversations: {str(e)}\n")
                continue  

        # ✅ NORMAL CHATBOT FLOW (NO MEMORY LOOKUP)
        try:
            async for chunk in app.astream(
                {"messages": user_conversations[user_id] + [HumanMessage(content=query)]},
                {
                    "thread_id": user_id,  
                    "checkpoint_ns": "chatbot",  
                    "checkpoint_id": f"{user_id}_{len(user_conversations[user_id])}"
                }
            ):
                chunk_content = getattr(chunk, "content", "")  # ✅ Avoid AttributeError

                if chunk_content:  
                    print(chunk_content, end="", flush=True)
                    response_text += chunk_content  

        except Exception as e:
            print(f"\n⚠️ Error during response streaming: {str(e)}")
            response_text = "Sorry, an error occurred while processing your request." 

        # ✅ Store only if valid response
        chatbot_response = AIMessage(content=response_text.strip() if response_text else "No response generated.")
        user_conversations[user_id].append(chatbot_response)
        save_to_db(user_id, query, chatbot_response.content)  


# Run the chatbot asynchronously
asyncio.run(chat())
