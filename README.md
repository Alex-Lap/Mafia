# Mafia LLM

A Python simulation of the social deduction game Mafia, where LLMs are assigned 
secret roles and play autonomously - reasoning, accusing, and voting each round 
using LangGraph to manage game state.

## Setup

1. Create and activate a virtual environment:
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate

2. Install dependencies:
   pip install -r requirements.txt

3. Create a .env file in the project root (make sure it's in .gitignore):
   ANTHROPIC_API_KEY=sk-ant-...

## Run

python main.py

## Roles

- Villager — identify and vote out the mafia
- Mafia — eliminate villagers without being caught
- Detective — secretly investigate one player per night
- Doctor — protect one player from elimination each night