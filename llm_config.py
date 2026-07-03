# llm_config.py
# ─────────────────────────────────────────────
# Central LLM switcher
# Students only change the DEFAULT_PROVIDER line
# ─────────────────────────────────────────────

import os
from pydantic import SecretStr
from dotenv import load_dotenv

load_dotenv()  # reads API keys from .env file

# ✏️ STUDENTS: Change this to your provider
# Options: "groq" | "gemini" | "claude" | "ollama"
DEFAULT_PROVIDER = "groq"


def _secret_from_env(name: str):
    value = os.getenv(name)
    return SecretStr(value) if value else None


def _required_secret_from_env(name: str) -> SecretStr:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return SecretStr(value)


def get_llm(provider: str = DEFAULT_PROVIDER):
    """
    Returns a LangChain-compatible chat model.
    Every agent calls this function to get the LLM.
    Change DEFAULT_PROVIDER above — all agents update automatically.
    """

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=_required_secret_from_env("GROQ_API_KEY"),
            temperature=0.3
        )

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            api_key=_required_secret_from_env("GEMINI_API_KEY"),
            temperature=0.3
        )

    elif provider == "claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model_name="claude-sonnet-4-6",
            api_key=_required_secret_from_env("ANTHROPIC_API_KEY"),
            temperature=0.3,
            timeout=30.0,
            stop=None
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model="llama3.1",
            temperature=0.3
        )

    else:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            "Choose from: groq, gemini, claude, ollama"
        )