# AI-Research-Agent
A FastAPI web app that plans a research workflow, runs tool-using agents (arXiv, Wikipedia, duckduckgo), and stores task state/results in Postgres. This repo includes a Docker setup that runs Postgres + the API in one container (for local/dev).

ai-research-agent/
├── src/
│   ├── __init__.py
│   ├── agents.py
│   ├── planning_agent.py
│   ├── research_tools.py
│   └── model_manager.py
├── templates/
│   └── index.html
├── static/
│   └── (CSS/JS files if any)
├── main.py
├── requirements.txt
├── Dockerfile            
├── docker-compose.yml      
├── .dockerignore          
├── .env.example           
└── README.md              
