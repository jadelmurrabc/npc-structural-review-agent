# NPC Structural Review Agent

Sub-Agent 1 for Qatar National Planning Council's Unidimensional Review system.

Evaluates strategy documents against 7 structural components (20 sub-components) and produces a scored markdown report.

## Setup
```
git clone https://github.com/jadelmurrabc/npc-structural-review-agent.git
cd npc-structural-review-agent
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
cp .env.example .env
# Edit .env with your Google Cloud project details
```

## Run on ADK Web
```
adk web
```
