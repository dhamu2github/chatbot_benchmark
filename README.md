# Charles Schwab Stock News Chatbot

## Project Overview
This is a sophisticated Streamlit-based chatbot designed to provide users with intelligent, context-aware responses about recent stock news using advanced semantic search and OpenAI's language model.

## Features
- Real-time stock news querying
- Semantic search functionality
- Prompt injection detection
- Interactive Streamlit UI
- AI-powered response generation

## Prerequisites
- Python 3.8+
- Streamlit
- OpenAI API Key
- Sentence Transformers
- Faiss
- python-dotenv

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/dhamu2github/chatbot_benchmark.git
cd chatbot_benchmark
```

### 2. Create Virtual Environment
- Create a Conda environment and activate it, assuming the machine has Anaconda installed.
```bash
conda create  -p <env name> python=<version>
conda activate <env name>
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Project Structure
```
stock-news-chatbot/
│
├── app.py               # Streamlit UI and main application logic
├── business_logic.py    # Core chatbot functionality
├── data/
│   └── stock_news.json  # News articles database
├── log/
│   └── chatbot.log      # Application logs
└── .env                 # Environment configuration
```

## Key Components
1. **Data Loading**: Loads stock news from JSON
2. **Semantic Search**: Uses Sentence Transformers and FAISS for efficient document retrieval
3. **Text Chunking**: Breaks articles into manageable chunks
4. **Prompt Injection Detection**: Prevents malicious query attempts
5. **OpenAI Response Generation**: Provides contextual answers

## Running the Application
```bash
streamlit run app.py
```

## Data Source

The application expects stock news data in JSON format at `data/stock_news.json` with the following structure:

```json
{
    "TICKER": [
        {
            "title": "Article Title",
            "full_text": "Article Content",
            "link": "Source URL"
        }
    ]
}
```

## Property Notice

This project is the exclusive property of Charles Schwab. Unauthorized use, reproduction, or distribution of this project without explicit permission is strictly prohibited.

## Technical Architecture

1. **Frontend Layer (app.py)**
   - Streamlit-based user interface
   - Chat history management
   - Response rendering and formatting

2. **Business Logic Layer (business_logic.py)**
   - News data loading and preprocessing
   - Semantic search implementation
   - LLM integration and response generation
   - Security checks and validations

3. **Data Layer**
   - Local JSON storage
   - Planned vector database integration

## Security Features
- Prompt injection detection
- Strict response generation based on available data
- Error handling and logging

## Logging
Logs are maintained in `log/chatbot.log` with timestamp, log level, and message details.

## Future Roadmap
- Support for additional LLM models (Gemini, LLAMA, Groq)
- Enhanced semantic search capabilities
- Improved prompt injection detection

## Potential Enhancements
- Caching mechanism for faster responses
- More granular error handling
- Advanced analytics on user queries

## Troubleshooting
- Ensure all dependencies are installed
- Verify OpenAI API key is valid
- Check network connectivity
- Review logs in `chatbot.log` for detailed error information

## Contributing
*Restricted to authorized Charles Schwab personnel*

## License
Proprietary - Charles Schwab

## Contact
For support or inquiries, contact the Charles Schwab development team.
