"""
DeepSeek API Client for Data Analysis Agent
"""

import os
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
import requests
import json

load_dotenv()


class DeepSeekClient:
    """Client for interacting with DeepSeek API"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        api_base: str = "https://api.deepseek.com",
    ):
        """
        Initialize DeepSeek client

        Args:
            api_key: DeepSeek API key (defaults to DEEPSEEK_API_KEY env var)
            model: Model name to use
            api_base: API base URL
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key not found. "
                "Set DEEPSEEK_API_KEY environment variable."
            )

        self.model = model
        self.api_base = api_base
        self.messages: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        """Add message to conversation history"""
        self.messages.append({"role": role, "content": content})

    def clear_history(self) -> None:
        """Clear conversation history"""
        self.messages = []

    def call(
        self,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """
        Send message to DeepSeek and get response

        Args:
            user_message: User's message
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response

        Returns:
            Assistant's response
        """
        self.add_message("user", user_message)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": self.messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()

            data = response.json()
            assistant_message = data["choices"][0]["message"]["content"]
            self.add_message("assistant", assistant_message)

            return assistant_message

        except requests.exceptions.RequestException as e:
            error_msg = f"DeepSeek API error: {str(e)}"
            print(error_msg)
            raise

    def analyze(
        self,
        context: str,
        question: str,
        temperature: float = 0.5,
    ) -> str:
        """
        Analyze data with given context

        Args:
            context: Data context/information
            question: Question to analyze
            temperature: Sampling temperature

        Returns:
            Analysis result
        """
        analysis_prompt = f"""
Context about the data:
{context}

Question: {question}

Please provide a detailed analysis based on the context provided.
"""
        return self.call(analysis_prompt, temperature=temperature)

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get full conversation history"""
        return self.messages.copy()