---
title: RAG Scouting Sport
emoji: 🏆
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.29.0
app_file: app_hf.py
pinned: false
---

# RAG Scouting Sport

Intelligent scouting tool using RAG (Retrieval-Augmented Generation) to answer natural-language questions about NBA and Premier League player statistics.

Live demo: https://huggingface.co/spaces/yanis-ghazi/rag-scouting

## Example questions

- "Which NBA player under 25 has the most assists this season?"
- "Find me an NBA point guard with more than 8 assists and fewer than 3 turnovers"
- "Who is the top scorer in the Premier League this season?"
- "Find me a PL defender under 23 with more than 5 goals"

## Architecture

User question

↓

Groq LLM extracts numeric filters

↓

ChromaDB retrieves by vector similarity

↓

Numeric filtering on metadata

↓

Groq LLM generates the final answer

## Tech stack

| Component | Technology |
|-----------|------------|
| LLM | Groq API (Llama 3.3 70B) |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector DB | ChromaDB |
| Interface | Gradio |
| Football data | FBref via soccerdata |
| Basketball data | Official NBA API |

## Data

- 508 NBA players: 2024/25 season (per-game stats: pts, reb, ast, stl, blk, tov, FG%, 3P%, FT%)
- 539 Premier League players: 2024/25 season (goals, assists, shots, minutes played)

## Installation

```bash
git clone https://github.com/yanis_ghazi/rag-scouting.git
cd rag-scouting

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

# Add your API key to .env
cp .env.example .env

python src/scraper.py
python src/preprocessor.py
python src/indexer.py
python app.py
```

## Project structure

rag-scouting/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── scraper.py
│   ├── preprocessor.py
│   ├── indexer.py
│   └── rag_engine.py
├── app.py
├── requirements.txt
