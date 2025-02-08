import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import trim_messages
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import SystemMessage
from langchain_community.document_loaders import PyPDFLoader  
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings  # ✅ Correct import
import asyncio
import wikipediaapi


# Load API Key from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
                    
# Debugging: Print API Key (remove this after testing)
if not api_key:
    raise ValueError("Missing OpenAI API Key. Check your .env file or set it manually.")
# print(api_key)

# Load documents
pdf_loader = PyPDFLoader("testRAG.pdf")  # For PDFs

# Split documents into smaller chunks for better retrieval
text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=50)
docs = text_splitter.split_documents(pdf_loader.load())

# Create a FAISS vector store
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(docs, embeddings)

# Save vector store to disk (optional for persistence)
vector_store.save_local("faiss_index")

# Set LangSmith API keys from .env
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

if not api_key:
    raise ValueError("Missing OpenAI API Key. Please set it in the environment.")

# Define model from chatGPT
model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=os.environ["OPENAI_API_KEY"])

# Store conversation memory for multiple users
user_conversations = {}  # Dictionary to store conversations by user

# Define trimmer
trimmer = trim_messages(
    max_tokens=1024,  # Adjust based on your needs
    strategy="last",  # Keep the most recent messages
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

def search_wikipedia(query):
    """Search Wikipedia only if 'Wikipedia' is mentioned in the query."""
    if "wikipedia" not in query.lower():
        return None  # Do nothing if "Wikipedia" isn't mentioned

    # Remove "Wikipedia" from query before searching
    clean_query = query.lower().replace("wikipedia", "").strip()

    wiki = wikipediaapi.Wikipedia(language = "en", user_agent="MyChatbot/1.0 (sofie.seilnacht@berkeley.edu)")
    page = wiki.page(clean_query)

    # Debugging: Check if Wikipedia is returning anything
    if page.exists():
        print("\n🌍 Wikipedia Search Found:\n", page.summary[:750])
        return page.summary[:750]  # Limit response length
    else:
        print("\n⚠️ No Wikipedia article found for:", clean_query)
        return "Sorry, no Wikipedia article found on that topic."
    
# Define the async function for model interaction
async def call_model(state: MessagesState):
    """Handles chatbot response asynchronously with RAG + Streaming."""

    # Trim messages to prevent token overload
    trimmed_messages = trimmer.invoke(state["messages"])

    # Extract latest user query
    latest_query = trimmed_messages[-1].content  
    
    # If "Wikipedia" is mentioned, fetch from Wikipedia
    wiki_answer = search_wikipedia(latest_query)
    if wiki_answer:  # If Wikipedia was searched, return its result
        return {"messages": state["messages"] + [AIMessage(content=wiki_answer)]}

    # Retrieve relevant documents from memory
    retrieved_docs = vector_store.similarity_search_with_score(latest_query, k=3)
    filtered_docs = [doc for doc, score in retrieved_docs if score > 0.3]
    retrieved_texts = "\n".join([doc.page_content for doc in filtered_docs])

    # Debugging: Print retrieved documents
    # if not retrieved_docs:
    #     print("\n⚠️ No matching documents found for query:", latest_query)
    # else:
    #     print("\n✅ Retrieved Documents for Query:", latest_query)
    #     for doc, score in retrieved_docs:
    #         print(f"\n🔹 Score: {score} - Content: {doc.page_content[:200]}...")

    # # ✅ Debug: Print retrieved knowledge inside function
    # print("\n🔍 Retrieved Documents for RAG:\n", retrieved_texts)

    # If documents are found, use them for the response
    if retrieved_texts:
        system_message = SystemMessage(content=f"Use the following knowledge:\n{retrieved_texts}")
        full_messages = [system_message] + trimmed_messages
    else:
        full_messages = trimmed_messages  # No documents? Just use OpenAI.

    # Stream chatbot response in real-time
    print("\n🤖 Chatbot:", end=" ", flush=True)
    response_text = ""

    async for chunk in model.astream(full_messages):  # Stream response chunks
        print(chunk.content, end="", flush=True)
        response_text += chunk.content  # Collect full response for storage

    print("\n")  # Newline after response finishes

    # Store final response in chat history
    response = AIMessage(content=response_text)

    return {"messages": trimmed_messages + [response]}  # Append response to chat history


# Define a new graph
workflow = StateGraph(state_schema=MessagesState)
# Define the (single) node in the graph
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")  # Define edges after nodes
# Add memory for state persistence
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Async function to run the chatbot interactively
async def chat():
    """Runs an interactive chatbot session in the terminal with user tracking."""
    
    config = {"configurable": {"thread_id": "abc123"}}

    # Ask for user ID only once at the start
    user_id = input("\nEnter your user ID: ").strip()
    if user_id.lower() == "bye":
        print("\n👋 Chatbot session ended. Goodbye!\n")
        return

    # Ensure this user has a conversation history
    if user_id not in user_conversations:
        user_conversations[user_id] = []  # Start fresh for new users

    print(f"\n💬 {user_id}, your chatbot session has started! Type 'bye' to stop.\n")

    while True:
        # Get user input with user ID formatting
        query = input(f"{user_id}: ").strip()
        if query.lower() == "bye":
            print(f"\n👋 Chatbot session ended for {user_id}. Goodbye!\n")
            break  # Exit the chat loop

        # Add user message to conversation history
        user_conversations[user_id].append(HumanMessage(content=query))

        # Get chatbot response
        output = await app.ainvoke({"messages": user_conversations[user_id]}, config)

        # Extract and print latest chatbot response
        chatbot_response = output["messages"][-1]
        # Append chatbot response to history
        user_conversations[user_id].append(chatbot_response)

# Run the chatbot asynchronously
asyncio.run(chat())