# RAG Chatbot

A corporate chatbot using Retrieval-Augmented Generation (RAG) with chat memory and document retrieval.

## Features

- Document-based Q&A using company policies
- Chat history with thread management
- Real-time streaming responses
- Reranking for better context retrieval

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Variables
Create `.env` file:
```
GEMINI_API_KEY=your_gemini_key
COHERE_API_KEY=your_cohere_key
PINECONE_API_KEY=your_pinecone_key
```

### 3. Run Locally
```bash
streamlit run app.py
```

## Deployment

### Streamlit Cloud
1. Push to GitHub (exclude `.env`)
2. Connect repository to Streamlit Cloud
3. Add secrets in dashboard:
   - `GEMINI_API_KEY`
   - `COHERE_API_KEY`
   - `PINECONE_API_KEY`

### Other Platforms
- Railway: Add environment variables
- Render: Set secrets in dashboard
- Heroku: Use config vars

## Files

- `app.py` - Main Streamlit interface
- `streamlit_chatbot.py` - RAG logic and LangGraph workflow
- `embedding.py` - Document embeddings setup
- `requirements.txt` - Dependencies
- `pdfs/` - Document storage

## Note

SQLite database resets on deployment restarts. Chat history is temporary on free hosting platforms.