import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
import asyncio

# Load API Key from .env file
load_dotenv()

# Set API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("Missing OpenAI API Key. Please set it in the environment.")

# Define model from chatGPT
model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=os.environ["OPENAI_API_KEY"])

# Store conversation memory for multiple users
user_conversations = {}  # Dictionary to store conversations by user

# Define the async function for model interaction
async def call_model(state: MessagesState):
    """Handles chatbot response asynchronously while maintaining memory."""
    messages = state["messages"]
    response = await model.ainvoke(messages)  # Get chatbot response
    return {"messages": messages + [response]}  # Append response to history

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