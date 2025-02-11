import os
import asyncio
import wikipediaapi
import numpy as np
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
model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"), streaming=True)

# Define a transformer to summarize past conversation data
summarizer = StrOutputParser()

# Load and process documents (FAISS Vector Store)
pdf_loader = PyPDFLoader("testRAG.pdf")
text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=50)
docs = text_splitter.split_documents(pdf_loader.load())

# Create FAISS vector store
embeddings = OpenAIEmbeddings()

if os.path.exists("faiss_index"):
    vector_store_docs = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
else:
    vector_store_docs = FAISS.from_documents(docs, embeddings)  # Create if missing
    vector_store_docs.save_local("faiss_index")  # Save it to disk

if os.path.exists("faiss_chat_memory"):
    vector_store_chat = FAISS.load_local("faiss_chat_memory", embeddings, allow_dangerous_deserialization=True)
else:
    # Create FAISS with a placeholder entry (FAISS requires at least one embedding to initialize)
    placeholder_text = ["This is a placeholder entry to initialize FAISS."]
    vector_store_chat = FAISS.from_texts(placeholder_text, embeddings)
    vector_store_chat.save_local("faiss_chat_memory")  # Save for future use


def add_documents_to_faiss(new_docs):
    """Adds new documents to the FAISS index and updates it."""
    global vector_store_docs  # Ensure we're updating the global FAISS index

    # Convert new docs into FAISS-compatible format
    new_vector_store = FAISS.from_documents(new_docs, embeddings)  

    # Merge new docs into existing FAISS index
    vector_store_docs.merge_from(new_vector_store)

    # Save updated FAISS index
    vector_store_docs.save_local("faiss_index")
    print("New documents added and FAISS index updated.")

def save_embedding_in_faiss(user_id, message):
    """Stores message embeddings in FAISS chat memory but prevents duplicates."""

    try:
        vector_store_chat = FAISS.load_local("faiss_chat_memory", embeddings, allow_dangerous_deserialization=True)
    except Exception:
        vector_store_chat = FAISS.from_texts(["This is a placeholder entry to initialize FAISS."], embeddings)

    # ✅ Check if the message is already stored
    query_embedding = embeddings.embed_query(message)
    existing_results = vector_store_chat.similarity_search_by_vector(query_embedding, k=3)

    for result in existing_results:
        if result.page_content.strip().lower() == message.strip().lower():
            return  # Don't save if it's already in FAISS

    # ✅ If it's a new message, store it
    vector_store_chat.add_texts([message], embeddings=[query_embedding], metadatas=[{"user_id": user_id}])
    vector_store_chat.save_local("faiss_chat_memory")

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
    """Retrieve relevant document chunks from FAISS (without reloading every time)."""
    search_results = vector_store_docs.similarity_search(query, k=1)  # Use global FAISS index
    if search_results:
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


def retrieve_relevant_chat(query):
    """Retrieves relevant past messages using FAISS chat memory and ensures persistence across chatbot restarts."""
    
    # ✅ Load FAISS chat memory from disk before querying
    try:
        vector_store_chat = FAISS.load_local("faiss_chat_memory", embeddings, allow_dangerous_deserialization=True)
    except Exception:
        return "💾 **Source: Chat Memory**\n\nNo relevant past conversations found."

    # ✅ Convert input query to vector
    query_embedding = embeddings.embed_query(query)

    # ✅ Retrieve top 3 most similar past chat messages
    results = vector_store_chat.similarity_search_by_vector(query_embedding, k=3)

    if results:
        retrieved_info = "\n".join([f"💾 **Stored Memory:** {doc.page_content}" for doc in results])
        return f"{retrieved_info}\n\n🔍 **New Query:** {query}"

    else:
        return "💾 **Source: Chat Memory**\n\nNo relevant past conversations found.\n\n🔍 **New Query:** " + query


# Wrap FAISS retrieval as a LangChain tool
tavily_tool = Tool(
    name="Web_Search",
    func=lambda query: search_tavily(query),
    description="Use this tool for retrieving real-time web information, especially for recent events, politics, or current facts.",)

faiss_tool = Tool(
    name="FAISS_Document_Search",
    func=lambda query: retrieve_from_faiss(query),
    description="Retrieve relevant information from stored documents.")

wikipedia_tool = Tool(
    name="Wikipedia_Search",
    func=lambda query: search_wikipedia(query),
    description="Retrieve factual summaries from Wikipedia.")

chat_memory_tool = Tool(
    name="Chat_Memory_Retrieval",
    func=lambda query: retrieve_relevant_chat(query),  # Ensure function receives query correctly
    description="Retrieve relevant past user conversations from FAISS to enhance responses.")

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

def save_to_db(user_id, message, response):
    """Stores user messages and responses in SQLite (without embeddings)."""
    
    cursor.execute("""
        INSERT INTO chat_history (user_id, message, response)
        VALUES (?, ?, ?)
    """, (user_id, message, response))
    conn.commit()

# async def call_model(state: MessagesState):
#     """Handles chatbot response asynchronously using the REACT agent with chat memory as a tool."""
#     trimmed_messages = trimmer.invoke(state["messages"])
#     latest_query = trimmed_messages[-1].content  

#     # ✅ Retrieve past chat memory (FAISS)
#     chat_memory_result = retrieve_relevant_chat(latest_query)
#     is_chat_memory_used = "💾 **Stored Memory:**" in chat_memory_result  

#     # ✅ Prepare memory context only if relevant
#     memory_context = ""
#     if is_chat_memory_used:
#         memory_context = f"💾 **Source: Chat Memory**\n\n{chat_memory_result}\n\n"
#         print(f"🧠 DEBUG: Passing FAISS memory into model:\n{memory_context}")  # Debugging

#     try:
#         # ✅ Step 1: Ask the agent (includes Tavily & Wikipedia)
#         agent_response = agent.invoke({"messages": [HumanMessage(content=memory_context + latest_query)]})

#         retrieved_info = ""
#         source_label = ""

#         if isinstance(agent_response, dict) and "messages" in agent_response:
#             messages_list = agent_response["messages"]
#             for msg in messages_list:
#                 if isinstance(msg, AIMessage):
#                     retrieved_info = msg.content.strip()

#         # ✅ Step 2: Determine Source (Web, FAISS, or Wikipedia)
#         if "🌐 **Source: Web Search (Tavily)**" in retrieved_info:
#             source_label = "🌐 Web Search:\n\n"
#         elif "🌍 **Source: Wikipedia**" in retrieved_info:
#             source_label = "🌍 Wikipedia:\n\n"
#         elif is_chat_memory_used:
#             source_label = "💾 Chat Memory:\n\n"

#         # ✅ Step 3: Format final response
#         if not retrieved_info:
#             print("⚠️ No AIMessage found. Using fallback response.")
#             retrieved_info = "I'm not sure how to respond to that."

#         final_response = f"{source_label}{retrieved_info}".strip()

#         return {"messages": trimmed_messages + [AIMessage(content=final_response)]}

#     except Exception as e:
#         print(f"\n❌ ERROR: `agent.invoke()` failed!\n{e}\n")
#         return {"messages": trimmed_messages + [AIMessage(content="Sorry, an error occurred.")]}

async def call_model(state: MessagesState):
    """Handles chatbot response asynchronously using the REACT agent with chat memory as a tool."""
    trimmed_messages = trimmer.invoke(state["messages"])
    latest_query = trimmed_messages[-1].content.lower()

    # ✅ Retrieve relevant past chat messages
    chat_memory_result = retrieve_relevant_chat(latest_query)
    is_chat_memory_used = "💾 **Stored Memory:**" in chat_memory_result

    # ✅ Set memory context only if it's useful
    memory_context = chat_memory_result if is_chat_memory_used else ""

    # ✅ Let the agent decide the tool to use
    try:
        agent_response = agent.invoke({"messages": [HumanMessage(content=memory_context + latest_query)]})

        retrieved_info = ""
        if isinstance(agent_response, dict) and "messages" in agent_response:
            messages_list = agent_response["messages"]
            for msg in messages_list:
                if isinstance(msg, AIMessage):
                    retrieved_info = msg.content

        if not retrieved_info:
            print("⚠️ No AIMessage found. Using fallback response.")
            retrieved_info = "I'm not sure how to respond to that."

    except Exception as e:
        print(f"\n❌ ERROR: `agent.invoke()` failed!\n{e}\n")
        retrieved_info = "Sorry, an error occurred."

    # ✅ Check which sources were actually used
    sources = []
    if is_chat_memory_used and any(text in retrieved_info for text in chat_memory_result.split("\n")):
        sources.append("💾Chat Memory: ")
    if "🌐 **Source: Web Search (Tavily)**" in retrieved_info:
        sources.append("🌐Web Search: ")
    if "📄 **Source: Retrieved from Documents**" in retrieved_info:
        sources.append("📄Documents: ")

    # ✅ Format the response correctly (Only include relevant sources)
    if sources:
        response_content = "\n\n".join(sources) + f"\n\n{retrieved_info}".strip()
    else:
        response_content = retrieved_info.strip()  # No unnecessary labels

    return {"messages": trimmed_messages + [AIMessage(content=response_content)]}


# Define the chatbot workflow
workflow = StateGraph(state_schema=MessagesState)
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

# Set up conversation memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

async def chat():
    """Runs an interactive chatbot session in the terminal with user tracking."""
    user_id = input("\nEnter your user ID: ").strip()
    if user_id.lower() == "bye":
        print("\n👋 Chatbot session ended. Goodbye!\n")
        return

    if user_id not in user_conversations:
        user_conversations[user_id] = []

    print(f"\n💬 {user_id}, your chatbot session has started! Type 'bye' to stop.\n")
    
    while True:
        query = input(f"{user_id}: ").strip()
        if not query:  
            print("⚠️ Please enter a message.")
            continue
        if query.lower() == "bye":
            print(f"\n👋 Chatbot session ended for {user_id}. Goodbye!\n")
            break

        # ✅ Save message in FAISS chat memory before response
        save_embedding_in_faiss(user_id, query)

        user_conversations[user_id].append(HumanMessage(content=query))  

        response_text = ""  

        print("\n🤖 Chatbot:", end=" ", flush=True)

        try:
            async for chunk in app.astream(
                {"messages": user_conversations[user_id] + [HumanMessage(content=query)]},
                {
                    "thread_id": user_id,  
                    "checkpoint_ns": "chatbot",  
                    "checkpoint_id": f"{user_id}_{len(user_conversations[user_id])}"
                }
            ):
                if "model" in chunk and "messages" in chunk["model"]:
                    ai_messages = chunk["model"]["messages"]  
                    if ai_messages and isinstance(ai_messages[-1], AIMessage):
                        chunk_content = ai_messages[-1].content.strip()
                    else:
                        chunk_content = ""  
                else:
                    chunk_content = ""  

                if chunk_content:
                    print(chunk_content, end="", flush=True)
                    response_text += chunk_content  

            print("\n")
        except Exception as e:
            print(f"\n⚠️ Error during response streaming: {str(e)}")
            response_text = "Sorry, an error occurred while processing your request."

        if not response_text.strip():
            response_text = "I didn't generate a response. Please try again."

        chatbot_response = AIMessage(content=response_text.strip())
        user_conversations[user_id].append(chatbot_response)
        save_to_db(user_id, query, response_text.strip())  


# Run the chatbot asynchronously
asyncio.run(chat())

