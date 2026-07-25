"""
Example usage of the Data Analysis Agent
Demonstrates various features and use cases
"""

import pandas as pd
from main import DataAnalysisAgent
from data_analyzer import DataAnalyzer


def create_sample_data():
    """Create sample dataset for testing"""
    import numpy as np

    np.random.seed(42)

    data = {
        "Date": pd.date_range("2024-01-01", periods=100),
        "Product": np.random.choice(["A", "B", "C"], 100),
        "Sales": np.random.randint(100, 1000, 100),
        "Quantity": np.random.randint(1, 50, 100),
        "Region": np.random.choice(["North", "South", "East", "West"], 100),
        "Customer_Rating": np.random.uniform(1, 5, 100),
    }

    df = pd.DataFrame(data)
    df.to_csv("sample_data.csv", index=False)
    print("✓ Sample data created: sample_data.csv")
    return "sample_data.csv"


def example_1_basic_analysis():
    """Example 1: Basic data analysis"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Data Analysis")
    print("="*70)

    # Create sample data
    data_file = create_sample_data()

    # Initialize agent
    agent = DataAnalysisAgent()
    agent.load_data(data_file)

    # Analyze
    queries = [
        "What are the top products by sales?",
        "Which region has the highest average rating?",
        "What is the trend in sales over time?",
    ]

    for query in queries:
        print(f"\n🤔 Query: {query}")
        response = agent.analyze_query(query)
        print(f"🤖 Response: {response}\n")


def example_2_data_analyzer():
    """Example 2: Using DataAnalyzer directly"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Direct Data Analyzer Usage")
    print("="*70)

    # Create sample data
    data_file = create_sample_data()

    # Initialize analyzer
    analyzer = DataAnalyzer(data_file)

    # Get summary
    print(analyzer.get_summary())

    # Filter data
    print("\n📊 Filtering: Sales > 500")
    filtered = analyzer.filter_data("Sales", ">", 500)
    print(f"Found {len(filtered)} records\n")

    # Group and aggregate
    print("📊 Group by Region - Average Sales:")
    result = analyzer.group_and_aggregate("Region", "Sales", "mean")
    print(result)

    # Create visualizations
    print("\n📈 Creating visualizations...")
    analyzer.plot_histogram("Sales", title="Sales Distribution")
    analyzer.plot_scatter("Quantity", "Sales", title="Quantity vs Sales")
    analyzer.plot_correlation_heatmap()


def example_3_batch_processing():
    """Example 3: Batch query processing"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Batch Query Processing")
    print("="*70)

    # Create sample data
    data_file = create_sample_data()

    # Initialize agent
    agent = DataAnalysisAgent()
    agent.load_data(data_file)

    # Batch queries
    queries = [
        "Summarize the dataset",
        "What are the key statistics?",
        "Identify any patterns or trends",
    ]

    results = agent.batch_analysis(queries)

    for query, response in results.items():
        print(f"\n❓ {query}")
        print(f"✓ {response}\n")


def example_4_export_results():
    """Example 4: Export results and conversation"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Export Results and Conversation")
    print("="*70)

    # Create sample data
    data_file = create_sample_data()

    # Initialize agent
    agent = DataAnalysisAgent()
    agent.load_data(data_file)

    # Run some analysis
    agent.analyze_query("What is the average sales per region?")
    agent.analyze_query("Which product has the highest rating?")

    # Export conversation
    agent.export_conversation("conversation_history.txt")
    print("✓ Conversation saved to conversation_history.txt")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num == "1":
            example_1_basic_analysis()
        elif example_num == "2":
            example_2_data_analyzer()
        elif example_num == "3":
            example_3_batch_processing()
        elif example_num == "4":
            example_4_export_results()
        else:
            print("Unknown example. Use: 1, 2, 3, or 4")
    else:
        print("Available examples:")
        print("  python example_usage.py 1  - Basic analysis")
        print("  python example_usage.py 2  - Data analyzer features")
        print("  python example_usage.py 3  - Batch processing")
        print("  python example_usage.py 4  - Export results")