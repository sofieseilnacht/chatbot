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

# Set up embeddings storage for chat history
def store_chat_embedding(message):
    embedding = embeddings.embed_query(message)  # Convert text to vector
    cursor.execute("UPDATE chat_history SET embedding = ? WHERE message = ?", (embedding, message))
    conn.commit()

# Set up Tavily Search
tavily = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True)

# Wrap Tavily as a LangChain tool
def search_tavily(query: str) -> str:
    """Search Tavily and return a summary of the topic."""
    search_results = tavily.invoke(query)

    if search_results:
        return f"🌐 **Source: Web Search (Tavily)**\n\n{search_results}"
    else:
        return "🌐 **Source: Web Search (Tavily)**\n\nNo relevant web results found."

def retrieve_from_faiss(query):
    """Retrieve relevant document chunks from FAISS."""
    vector_store = FAISS.load_local("faiss_index", embeddings)
    search_results = vector_store.similarity_search(query, k=3)

    if  search_results:
        return f"📄 **Source: Retrieved from Documents**\n\n" + "\n".join([doc.page_content for doc in search_results])
    else:
        return "📄 **Source: Retrieved from Documents**\n\nNo relevant documents found."

# Wikipedia Search (Only when explicitly mentioned)
def search_wikipedia(query: str) -> str:
    """Search Wikipedia and return a summary of the topic."""
    wiki = wikipediaapi.Wikipedia(language="en", user_agent="MyChatbot/1.0")
    page = wiki.page(query)

    if page.exists():
        return f"🌍 **Source: Wikipedia**\n\n{page.summary[:750]}"
    else:
        return "🌍 **Source: Wikipedia**\n\nNo Wikipedia article found."

def retrieve_relevant_chat(query: str) -> str:
    """Retrieve the most relevant past chat logs using vector similarity search."""
    query_embedding = embeddings.embed_query(query)  # Convert query to vector
    
    # Retrieve top 3 relevant past chats using FAISS or a vector database
    results = vector_store.similarity_search_by_vector(query_embedding, k=3)

    if results:
        return f"💾 **Source: Chat Memory**\n\n" + "\n".join([msg.page_content for msg in results])
    else:
        return "💾 **Source: Chat Memory**\n\nNo relevant past information found."

def auto_retrieve_past_info(query):
    """Automatically retrieve past relevant chat history."""
    past_info = retrieve_relevant_chat(query)  # Fetch relevant past logs
    
    if past_info:
        return f"💾 **Using past conversations:**\n{past_info}\n\n{query}"
    else:
        return query  # If no past data is relevant, just use the query

# Wrap FAISS retrieval as a LangChain tool
tavily_tool = Tool(
    name="Web Search",
    func=lambda query: search_tavily(query),
    description="Retrieve real-time web information.")

faiss_tool = Tool(
    name="FAISS Document Search",
    func=lambda query: retrieve_from_faiss(query),
    description="Retrieve relevant information from stored documents.")

wikipedia_tool = Tool(
    name="Wikipedia Search",
    func=lambda query: search_wikipedia(query),
    description="Retrieve factual summaries from Wikipedia.")

chat_memory_tool = Tool(
    name="Chat Memory Retrieval",
    func=retrieve_relevant_chat,
    description="Retrieve relevant past user conversations to enhance responses.")

# Create REACT Agent (Decides which tool to use)
agent = create_react_agent(
    model,
    tools=[tavily_tool, faiss_tool, wikipedia_tool, chat_memory_tool])

# Define conversation memory per user
user_conversations = {}

# Define Trimmer (to prevent token overload)
trimmer = trim_messages(
    max_tokens=1024,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",)

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect("chat_memory.db")
cursor = conn.cursor()

# Create a table to store chat history
cursor.execute("""
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    message TEXT,
    response TEXT)""")
conn.commit()

# def get_recent_past_conversations(user_id, limit=10):
#     """Fetch the last 'limit' messages from the database, excluding recall requests."""
    
#     # Debug: Print the query being executed
#     print(f"Retrieving last {limit} messages for user: {user_id}")

#     cursor.execute("""
#         SELECT message, response FROM chat_history
#         WHERE user_id = ? 
#         ORDER BY id DESC
#         LIMIT ?
#     """, (user_id, int(limit)))  # Ensure limit is an integer
    
#     results = cursor.fetchall()

#     # 🔹 Filter out messages where the user input contains "recall" (case-insensitive)
#     filtered_results = [msg for msg in results if "recall" not in msg[0].lower()]

#     # Get only the last `limit` messages **after filtering** to ensure correct count
#     final_results = filtered_results[-limit:]

#     return final_results


def save_to_db(user_id, message, response):
    """Stores user messages, response, and embeddings in SQLite."""
    
    # Insert message & response into the table first (embedding set as NULL initially)
    cursor.execute("""
        INSERT INTO chat_history (user_id, message, response, embedding)
        VALUES (?, ?, ?, NULL)
    """, (user_id, message, response))
    conn.commit()

    # Generate embedding for the message
    embedding = embeddings.embed_query(message)  # Convert text to vector

    # Store the embedding in the correct row
    cursor.execute("""
        UPDATE chat_history SET embedding = ? WHERE user_id = ? AND message = ?
    """, (embedding, user_id, message))
    conn.commit()

async def call_model(state: MessagesState):
    """Handles chatbot response asynchronously using the REACT agent with chat memory as a tool."""
    
    trimmed_messages = trimmer.invoke(state["messages"])
    latest_query = trimmed_messages[-1].content  

    # Invoke the agent (it decides if retrieval is needed)
    agent_response = agent.invoke([HumanMessage(content=latest_query)])

    # Determine if retrieval happened
    if isinstance(agent_response, str):
        retrieved_info = agent_response  # Normal chat response
        source_text = None  # No retrieval source
    else:
        retrieved_info = agent_response.get("content", "No relevant information found.")
        source_text = agent_response.get("source", None)  # Set to None if not found

    # Only print retrieval info if a tool was used
    if source_text:
        print("\n🔍 Selecting the best retrieval method...\n", flush=True)
        print(f"\n{source_text}\n", flush=True)

    # Send response to LLM (whether it’s retrieval-based or normal conversation)
    async for chunk in model.astream(trimmed_messages + [HumanMessage(content=retrieved_info)]):
        print(chunk.content, end="", flush=True)

# async def call_model(state: MessagesState):
#     """Handles chatbot response asynchronously using the REACT agent with chat memory as a tool."""

#     trimmed_messages = trimmer.invoke(state["messages"])
#     latest_query = trimmed_messages[-1].content  

#     print("\n🔍 Selecting the best retrieval method...\n", flush=True)

#     # Invoke the agent, which decides if past chat retrieval is needed
#     agent_response = agent.invoke([HumanMessage(content=latest_query)])

#     # Extract the tool's response
#     if isinstance(agent_response, str):
#         retrieved_info = agent_response  # If response is just a string
#         source_text = "🔍 Just chatting"
#     else:
#         retrieved_info = agent_response.get("content", "No relevant information found.")
#         source_text = agent_response.get("source", "🔍 **Source: Chat Memory**")

#     # Print source before chatbot response
#     print(f"\n{source_text}\n", flush=True)

#     # Send retrieved info to LLM for final response
#     print("\n🤖 Chatbot:", end=" ", flush=True)
#     async for chunk in model.astream(trimmed_messages + [HumanMessage(content=retrieved_info)]):
#         print(chunk.content, end="", flush=True)

# async def call_model(state: MessagesState):
#     """Handles chatbot response asynchronously using the REACT agent with real-time streaming."""

#     # Trim messages to prevent token overload
#     trimmed_messages = trimmer.invoke(state["messages"])
    
#     # Extract latest user query
#     latest_query = trimmed_messages[-1].content  

#     # Handle Recall Requests ONLY When Necessary
#     if any(keyword in latest_query.lower() for keyword in ["recall", "remember", "look back"]):
#         user_id = state.get("user_id", "default_user")  # Get user ID
#         past_chats = get_recent_past_conversations(user_id, limit=10)  # Only fetch last 10 messages

#         if past_chats:
#             # Convert past messages into a formatted summary
#             chat_history_text = "\n".join([f"User: {msg} | Bot: {resp}" for msg, resp in past_chats])
#             summary = summarizer.parse(f"Summarize user past interactions and provide a direct answer:\n{chat_history_text}")
            
#             # Modify query to reflect the recall summary, ensuring a **direct answer**
#             query = f"Based on this summary: {summary}, {latest_query}"
#         else:
#             return {"messages": trimmed_messages + [AIMessage(content="I don't seem to have past records.")]}  # Early return if no records exist

#     else:
#         # If no recall is requested, process query as usual
#         query = latest_query  # Keep the original input

#     # Streaming Chatbot Response
#     print("\n🤖 Chatbot:", end=" ", flush=True)
#     response_text = ""

#     async for chunk in model.astream(trimmed_messages + [HumanMessage(content=query)]):
#         if hasattr(chunk, "content"):  # Check if 'chunk' has a 'content' attribute
#             chunk_content = chunk.content  # Directly access 'content'
#         else:
#             chunk_content = ""

#         if chunk_content:
#             print(chunk_content, end="", flush=True)  # Stream response
#             response_text += chunk_content  # Collect response text


#     print("\n")  # Newline after response finishes

#     # Store final response in chat history & return
#     response = AIMessage(content=response_text.strip())

#     return {"messages": trimmed_messages + [response]}  # Append response to chat history

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
        response_text = ""  # Always initialize before using

        # Check if query requires recall (only then access database) 
        if any(keyword in query.lower() for keyword in ["recall", "remember"]):
            try:
                past_chats = get_recent_past_conversations(user_id, limit=10)  

                if past_chats:
                    chat_history_text = "\n".join([f"User: {msg} | Bot: {resp}" for msg, resp in past_chats])

                    # Generate a **direct answer** based on past conversation
                    recall_prompt = f"""
                    The user wants to recall past conversations. 
                    Here are the last few exchanges:
                    
                    {chat_history_text}

                    User's latest question: "{query}"

                    🔹 Answer clearly and concisely, referencing past interactions. 
                    🔹 If the user's query is about their age, location, or facts about themselves, return an exact answer.
                    🔹 DO NOT summarize vaguely—give a factual answer.
                    """

                    response = model.invoke([HumanMessage(content=recall_prompt)])  # Force direct response
                    response_text = response.content.strip()

                    print(f"\n🔍 Recall Triggered. Using past memory for response.\n")
                    print(f"\n🤖 Chatbot: {response_text}\n")  # Print response immediately

                    chatbot_response = AIMessage(content=response_text)
                    user_conversations[user_id].append(chatbot_response)
                    save_to_db(user_id, query, chatbot_response.content)  
                    continue  # Skip normal chatbot flow after recall

                else:
                    print("\n🤖 Chatbot: I don't seem to have past records.\n")
                    continue  

            except Exception as e:
                print(f"\n⚠️ Error fetching past conversations: {str(e)}\n")
                continue  

        # NORMAL CHATBOT FLOW (NO MEMORY LOOKUP)
        try:
            async for chunk in app.astream(
                {"messages": user_conversations[user_id] + [HumanMessage(content=query)]},
                {
                    "thread_id": user_id,  
                    "checkpoint_ns": "chatbot",  
                    "checkpoint_id": f"{user_id}_{len(user_conversations[user_id])}"
                }
            ):
                chunk_content = getattr(chunk, "content", "")  # Avoid AttributeError

                if chunk_content:  
                    print(chunk_content, end="", flush=True)
                    response_text += chunk_content  

        except Exception as e:
            print(f"\n⚠️ Error during response streaming: {str(e)}")
            response_text = "Sorry, an error occurred while processing your request." 

        # Store only if valid response
        chatbot_response = AIMessage(content=response_text.strip() if response_text else "No response generated.")
        user_conversations[user_id].append(chatbot_response)
        save_to_db(user_id, query, chatbot_response.content)  


# Run the chatbot asynchronously
asyncio.run(chat())
