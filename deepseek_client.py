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
        api_base: str = "https://api.deepseek.com/v1",
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
            "stream": False,
        }

        try:
            # Debug info
            if os.getenv("DEBUG") == "True":
                print(f"\n[DEBUG] API Base: {self.api_base}")
                print(f"[DEBUG] Model: {self.model}")
                print(f"[DEBUG] Payload: {json.dumps(payload, indent=2)}")

            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )

            # Debug response
            if os.getenv("DEBUG") == "True":
                print(f"[DEBUG] Status Code: {response.status_code}")
                print(f"[DEBUG] Response: {response.text}")

            # Check for errors
            if response.status_code != 200:
                error_detail = response.text
                print(f"❌ DeepSeek API error ({response.status_code}): {error_detail}")
                raise requests.exceptions.RequestException(
                    f"Status {response.status_code}: {error_detail}"
                )

            data = response.json()
            
            # Handle different response formats
            if "choices" not in data or len(data["choices"]) == 0:
                print(f"❌ Unexpected API response format: {data}")
                raise ValueError("No choices in API response")

            assistant_message = data["choices"][0]["message"]["content"]
            self.add_message("assistant", assistant_message)

            return assistant_message

        except requests.exceptions.Timeout:
            error_msg = "❌ DeepSeek API timeout: Request took too long"
            print(error_msg)
            raise
        except requests.exceptions.ConnectionError:
            error_msg = "❌ DeepSeek API connection error: Cannot reach API"
            print(error_msg)
            raise
        except requests.exceptions.RequestException as e:
            error_msg = f"❌ DeepSeek API error: {str(e)}"
            print(error_msg)
            raise
        except json.JSONDecodeError:
            error_msg = "❌ Invalid JSON response from DeepSeek API"
            print(error_msg)
            raise
        except KeyError as e:
            error_msg = f"❌ Unexpected response structure: {str(e)}"
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
        analysis_prompt = f"""Context about the data:
{context}

Question: {question}

Please provide a detailed analysis based on the context provided. Answer in the same language as the question."""
        return self.call(analysis_prompt, temperature=temperature)

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get full conversation history"""
        return self.messages.copy()

    def test_connection(self) -> bool:
        """Test DeepSeek API connection"""
        try:
            print("🔌 Testing DeepSeek API connection...")
            test_message = "Hello, this is a test message. Please respond briefly."
            self.add_message("user", test_message)

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": test_message}],
                "temperature": 0.5,
                "max_tokens": 100,
                "stream": False,
            }

            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )

            if response.status_code == 200:
                print("✅ DeepSeek API connection successful!")
                return True
            else:
                print(f"❌ DeepSeek API returned status {response.status_code}")
                print(f"Response: {response.text}")
                return False

        except Exception as e:
            print(f"❌ Connection test failed: {str(e)}")
            return False
