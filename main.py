"""
Main AI Agent for Data Analysis using DeepSeek
Interactive agent for analyzing data and answering questions
"""

import os
from typing import Optional
from pathlib import Path
from deepseek_client import DeepSeekClient
from data_analyzer import DataAnalyzer


class DataAnalysisAgent:
    """Main AI Agent for data analysis"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the agent

        Args:
            api_key: DeepSeek API key
        """
        self.deepseek = DeepSeekClient(api_key=api_key)
        self.analyzer: Optional[DataAnalyzer] = None
        self.data_loaded = False

    def load_data(self, file_path: str) -> bool:
        """Load data file"""
        if not os.path.exists(file_path):
            print(f"✗ File not found: {file_path}")
            return False

        self.analyzer = DataAnalyzer(file_path)
        self.data_loaded = True
        return True

    def analyze_query(self, query: str) -> str:
        """
        Analyze user query using DeepSeek

        Args:
            query: User's analysis question

        Returns:
            AI analysis response
        """
        if not self.data_loaded or self.analyzer is None:
            return "Please load data first using load_data()"

        # Prepare context with data summary
        data_summary = self.analyzer.get_summary()
        context = f"""
I have loaded a dataset with the following characteristics:
{data_summary}

User's question: {query}

Please provide a helpful analysis based on the data summary.
"""

        try:
            response = self.deepseek.call(
                context,
                temperature=0.7,
                max_tokens=1500,
            )
            return response
        except Exception as e:
            return f"Error during analysis: {str(e)}"

    def interactive_session(self) -> None:
        """Run interactive analysis session"""
        print("\n" + "="*60)
        print("🤖 AI Data Analysis Agent (DeepSeek)")
        print("="*60)

        # Load data
        print("\n📁 Step 1: Load your data file")
        while True:
            file_path = input("Enter path to data file (CSV/Excel/JSON): ").strip()
            if self.load_data(file_path):
                print("✓ Data loaded successfully!")
                break
            else:
                print("✗ Failed to load data. Try again.")

        print("\n📊 Data Summary:")
        print(self.analyzer.get_summary())

        # Interactive analysis loop
        print("\n💬 Step 2: Ask questions about your data")
        print("(Type 'quit' to exit, 'summary' for data summary)")
        print("-" * 60)

        while True:
            user_input = input("\n🤔 Your question: ").strip()

            if user_input.lower() == "quit":
                print("\n👋 Thank you for using the Data Analysis Agent!")
                break
            elif user_input.lower() == "summary":
                print(self.analyzer.get_summary())
            elif user_input:
                print("\n🔄 Analyzing...")
                response = self.analyze_query(user_input)
                print(f"\n🤖 Analysis:\n{response}")
            else:
                print("Please enter a question.")

    def batch_analysis(self, queries: list) -> dict:
        """
        Analyze multiple queries at once

        Args:
            queries: List of queries to analyze

        Returns:
            Dictionary with queries and responses
        """
        results = {}
        for i, query in enumerate(queries, 1):
            print(f"Analyzing query {i}/{len(queries)}...")
            results[query] = self.analyze_query(query)
        return results

    def clear_conversation(self) -> None:
        """Clear conversation history"""
        self.deepseek.clear_history()
        print("✓ Conversation history cleared")

    def export_conversation(self, file_path: str) -> bool:
        """
        Export conversation history to file

        Args:
            file_path: Path to save conversation

        Returns:
            True if successful
        """
        try:
            history = self.deepseek.get_conversation_history()
            with open(file_path, "w", encoding="utf-8") as f:
                for i, msg in enumerate(history, 1):
                    f.write(f"\n{'='*60}\n")
                    f.write(f"Message {i} - Role: {msg['role'].upper()}\n")
                    f.write(f"{'='*60}\n")
                    f.write(msg["content"] + "\n")
            print(f"✓ Conversation exported to {file_path}")
            return True
        except Exception as e:
            print(f"✗ Error exporting conversation: {str(e)}")
            return False


def main():
    """Main entry point"""
    import sys

    # Create agent
    agent = DataAnalysisAgent()

    # Example: Load sample data and run analysis
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        if agent.load_data(data_file):
            print(f"✓ Data loaded: {data_file}")
            # Run interactive session
            agent.interactive_session()
        else:
            print(f"✗ Could not load data file: {data_file}")
    else:
        # Run interactive session
        agent.interactive_session()


if __name__ == "__main__":
    main()