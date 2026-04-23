import os
from dotenv import load_dotenv

load_dotenv()

def get_llm(provider="openai"):

    if provider == "openai":
        from langchain.chat_models import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini")

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-pro")

    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model="llama3-8b-8192")

    elif provider == "bedrock":
        import boto3
        from langchain_community.chat_models import BedrockChat

        client = boto3.client("bedrock-runtime")
        return BedrockChat(
            client=client,
            model_id="anthropic.claude-v2"
        )

    else:
        raise ValueError("Unsupported LLM provider")