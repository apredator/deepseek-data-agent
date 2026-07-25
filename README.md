# 🤖 DeepSeek Data Analysis Agent

An AI-powered data analysis agent using DeepSeek API and Python. Analyze your data, ask questions, and get intelligent insights powered by DeepSeek's language model.

## ✨ Features

- 📊 **Data Loading**: Support for CSV, Excel, and JSON files
- 🤖 **AI Analysis**: Intelligent queries using DeepSeek API
- 📈 **Visualization**: Create histograms, scatter plots, and correlation heatmaps
- 💬 **Interactive Mode**: Chat with the AI about your data
- 📋 **Batch Processing**: Analyze multiple queries at once
- 💾 **Export Results**: Save conversations and analysis results
- 📝 **Data Aggregation**: Group by columns and calculate statistics

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/apredator/deepseek-data-agent.git
cd deepseek-data-agent
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up API Key
```bash
cp .env.example .env
# Edit .env and add your DeepSeek API key
```

### 4. Run the Agent
```bash
python main.py
```

## 📚 Usage

### Interactive Mode
```bash
python main.py
# Follow prompts to load data and ask questions
```

### Programmatic Usage
```python
from main import DataAnalysisAgent

# Create agent
agent = DataAnalysisAgent()

# Load data
agent.load_data("your_data.csv")

# Ask questions
response = agent.analyze_query("What is the average sales by region?")
print(response)
```

### Using DataAnalyzer Directly
```python
from data_analyzer import DataAnalyzer

# Load data
analyzer = DataAnalyzer("your_data.csv")

# Get summary
print(analyzer.get_summary())

# Filter data
filtered = analyzer.filter_data("Sales", ">", 500)

# Create visualizations
analyzer.plot_histogram("Sales")
analyzer.plot_scatter("Quantity", "Sales")
analyzer.plot_correlation_heatmap()
```

## 📝 Examples

Run the example scripts:
```bash
# Example 1: Basic analysis
python example_usage.py 1

# Example 2: Data analyzer features
python example_usage.py 2

# Example 3: Batch processing
python example_usage.py 3

# Example 4: Export results
python example_usage.py 4
```

## 📁 Project Structure

```
deepseek-data-agent/
├── main.py                 # Main agent application
├── deepseek_client.py      # DeepSeek API client
├── data_analyzer.py        # Data analysis module
├── example_usage.py        # Usage examples
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
└── README.md              # This file
```

## 🔧 Configuration

### Environment Variables (.env)
```
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_API_BASE=https://api.deepseek.com
DEBUG=False
```

### Supported File Formats
- CSV (.csv)
- Excel (.xlsx, .xls)
- JSON (.json)

## 💡 API Methods

### DataAnalysisAgent
- `load_data(file_path)` - Load data file
- `analyze_query(query)` - Analyze user query using AI
- `batch_analysis(queries)` - Analyze multiple queries
- `interactive_session()` - Run interactive chat session
- `export_conversation(file_path)` - Export conversation history
- `clear_conversation()` - Clear conversation history

### DataAnalyzer
- `load_data(file_path)` - Load data
- `get_summary()` - Get data summary
- `get_detailed_stats()` - Get detailed statistics
- `filter_data(column, operator, value)` - Filter data
- `group_and_aggregate(group_by, agg_column, agg_func)` - Group and aggregate
- `plot_histogram(column, ...)` - Create histogram
- `plot_scatter(x_column, y_column, ...)` - Create scatter plot
- `plot_correlation_heatmap()` - Create correlation heatmap
- `export_results(output_path)` - Export results to CSV

## 🔑 Getting DeepSeek API Key

1. Visit [DeepSeek API](https://www.deepseek.com)
2. Sign up for an account
3. Generate API key from dashboard
4. Add to `.env` file

## 📊 Example Analysis Workflow

```python
from main import DataAnalysisAgent

# Initialize agent
agent = DataAnalysisAgent()

# Load your dataset
agent.load_data("sales_data.csv")

# Ask analytical questions
print(agent.analyze_query("What are the top 5 customers by revenue?"))
print(agent.analyze_query("Show the sales trend over time"))
print(agent.analyze_query("Which products have the best margins?"))

# Export conversation
agent.export_conversation("analysis_report.txt")
```

## 🐛 Troubleshooting

### API Key Error
```
ValueError: DeepSeek API key not found
```
**Solution**: Make sure DEEPSEEK_API_KEY is set in your `.env` file

### File Not Found
```
File not found: your_data.csv
```
**Solution**: Verify file path and ensure file exists

### Visualization Issues
If plots don't display, try:
```python
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
```

## 📈 Performance Tips

- Keep datasets under 100MB for best performance
- Use CSV format for faster loading
- Filter data before analysis to reduce query time
- Batch similar queries together

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 💬 Support

For issues and questions:
1. Check the troubleshooting section
2. Review examples in `example_usage.py`
3. Open an issue on GitHub

## 🎯 Roadmap

- [ ] Support for database connections
- [ ] Advanced ML model integration
- [ ] Real-time data streaming
- [ ] Web interface dashboard
- [ ] Multi-language support
- [ ] GPU acceleration for large datasets

---

**Made with ❤️ for data analysts and AI enthusiasts**