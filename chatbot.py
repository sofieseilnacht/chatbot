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
import asyncio

# Load API Key from .env file
load_dotenv()

# Set API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Set LangSmith API keys from .env
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

if not api_key:
    raise ValueError("Missing OpenAI API Key. Please set it in the environment.")

# Define model from chatGPT
model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=os.environ["OPENAI_API_KEY"])

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)



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

# Define the async function for model interaction
async def call_model(state: MessagesState):
    """Handles chatbot response asynchronously with RAG (retrieval)."""

    # Trim messages to prevent overload
    trimmed_messages = trimmer.invoke(state["messages"])

    # Extract latest user query
    latest_query = trimmed_messages[-1].content  

    # Retrieve relevant documents from memory
    retrieved_docs = vector_store.similarity_search(latest_query, k=3)
    filtered_docs = [doc for doc, score in retrieved_docs if score > 0.7]
    retrieved_texts = "\n".join([doc.page_content for doc in filtered_docs])

    # Add retrieved knowledge to the model prompt
    system_message = SystemMessage(content=f"Use the following knowledge:\n{retrieved_texts}")
    full_messages = [system_message] + trimmed_messages

    # Stream chatbot response (Real-time output)
    response_text = ""
    async for chunk in model.astream(full_messages):  # Stream response chunks
        response_text += chunk.content  # Build response as it streams

    # Store final response
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
        print(f"\n🤖 {user_id}, Chatbot: {chatbot_response.content}\n")

        # Append chatbot response to history
        user_conversations[user_id].append(chatbot_response)

# Run the chatbot asynchronously
asyncio.run(chat())