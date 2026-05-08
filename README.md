# UdaPlay — AI Game Research Agent

An AI agent that answers questions about the video game industry using a two-tier information retrieval system:

1. **Local knowledge base** — ChromaDB vector store with 15 game documents, searched via RAG.
2. **Web fallback** — Tavily API live search, triggered only when the local DB is insufficient.

## Project structure

```
building_agent_final_project/
├── final_project.py            ← runnable Python script
├── building_agent_final.ipynb  ← step-by-step notebook guide
├── lib/                        ← helper library (LLM, Agent, VectorStore, …)
├── games/                      ← 15 game JSON files
├── requirements.txt
├── .env.example                ← copy to .env and fill in your keys
└── .gitignore
```

## Quick start

```bash
# 1. Clone the repo
git clone <repo-url>
cd building_agent_final_project

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up API keys
cp .env.example .env
# Edit .env and fill in OPENAI_API_KEY and TAVILY_API_KEY

# 5. Run the agent
python final_project.py
```

## Notebook guide

Open `building_agent_final.ipynb` in Jupyter or VS Code.  
It walks through every implementation step in order — fix three library bugs, build the vector DB, implement the three tools, assemble the agent, and run test queries.

## How the agent works

```
User question
    │
    ▼
retrieve_game(query)          ← semantic search in ChromaDB
    │
    ▼
evaluate_retrieval(question, retrieved_docs)   ← LLM-as-judge
    │
    ├─ useful=True  ──► answer from internal DB
    │
    └─ useful=False ──► game_web_search(question) ──► answer from web
```

## Agent tools

| Tool | Description |
|------|-------------|
| `retrieve_game` | Embeds the query and searches the ChromaDB collection |
| `evaluate_retrieval` | Uses `gpt-4o-mini` + Pydantic structured output to judge retrieval quality |
| `game_web_search` | Calls the Tavily API for live web results |

## Requirements

- Python 3.11+
- OpenAI API key (Vocareum-compatible endpoint)
- Tavily API key — [get one free at tavily.com](https://tavily.com)
