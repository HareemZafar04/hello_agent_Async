import os
import asyncio
from dotenv import load_dotenv
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, Runner


load_dotenv()


gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    raise ValueError("Please set the GEMINI_API_KEY environment variable.")


client = AsyncOpenAI(
    api_key=gemini_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


chat_model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

run_config = RunConfig(
    model=chat_model,
    model_provider=client,
    tracing_disabled=True
)

async def start_chat():
    
    chatbot = Agent(
        name="SmartBuddy",
        instructions=(
            "You are a friendly and knowledgeable AI companion. "
            "Respond to user questions clearly, concisely, and politely. "
            "Keep answers short unless the user requests elaboration. "
            "If asked for examples or templates, provide brief, relevant ones. "
            "Always offer to provide more details if needed."
        ),
        model=chat_model
    )

    print("Hello and welcome to SmartBuddy! Type 'stop' or 'end' to exit.\n")

    
    while True:
        user_message = input("Your question: ")
        if user_message.lower() in ["stop", "end"]:
            print("SmartBuddy: Thanks for chatting! Come back anytime!")
            break

       
        response = await Runner.run(chatbot, user_message, run_config=run_config)

       
        print("SmartBuddy:", response.final_output)


if __name__ == "__main__":
    asyncio.run(start_chat())