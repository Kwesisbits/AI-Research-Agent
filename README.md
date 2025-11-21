##  AI Research Agent with Multi-Agent System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a393.svg)](https://fastapi.tiangolo.com/)

A multi-agent research assistant that autonomously conducts research, synthesizes findings, and generates publication-quality reports. Powered entirely by open-source LLMs from Hugging Face.

![Demo](https://via.placeholder.com/800x400.png?text=AI+Research+Agent+Demo)

## Features

- **Multi-Agent Collaboration**: Four specialized agents (Planner, Researcher, Writer, Editor) work together
- **100% Open Source**: Runs on Hugging Face models (Mistral, Llama, Mixtral) - no OpenAI required
- **Comprehensive Research**: Integrates DuckDuckGo search, arXiv papers, and Wikipedia
- **Real-Time Progress**: Watch agents work with live updates
- **Production Ready**: Dockerized FastAPI backend with SQLite/PostgreSQL support

## Quick Start

```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/ai-research-agent.git
cd ai-research-agent

# Using Docker (recommended)
docker-compose up --build

# Or local setup
pip install -r requirements.txt
uvicorn main:app --reload
```

Visit `http://localhost:8000` and start researching!

## Architecture

```
User Query → Planner Agent (Mistral-7B)
              ↓
          Research Agent (Hermes-2-Pro-Llama-3-8B)
              ├─→ DuckDuckGo Search
              ├─→ arXiv Papers
              └─→ Wikipedia
              ↓
          Writer Agent (Mixtral-8x7B)
              ↓
          Editor Agent (Llama-3-8B)
              ↓
          Final Report (Markdown)
```

## Project Structure

```
ai-research-agent/
├── src/
│   ├── model_manager.py      # HF model loading & inference
│   ├── agents.py              # Research/Writer/Editor agents
│   ├── planning_agent.py      # Planner & orchestrator
│   └── research_tools.py      # DuckDuckGo/arXiv/Wikipedia
├── templates/
│   └── index.html             # Web UI
├── main.py                    # FastAPI app
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Environment Variables

Create a `.env` file:

```bash
# Optional: For private Hugging Face models
HF_TOKEN=hf_xxxxxxxxxxxxx

# Database (SQLite by default)
DATABASE_URL=sqlite:///./data/research_agent.db

# Contact for arXiv API
CONTACT_EMAIL=your.email@example.com
```

## API Usage

```python
import requests

# Start research
response = requests.post(
    "http://localhost:8000/generate_report",
    json={"prompt": "Recent advances in transformers"}
)
task_id = response.json()["task_id"]

# Poll progress
progress = requests.get(f"http://localhost:8000/task_progress/{task_id}")

# Get final report
result = requests.get(f"http://localhost:8000/task_status/{task_id}")
print(result.json()["result"]["html_report"])
```

## Models Used

| Agent | Model | Size | Purpose |
|-------|-------|------|---------|
| Planner | Mistral-7B-Instruct | 7B | Planning & coordination |
| Research | Hermes-2-Pro-Llama-3-8B | 8B | Tool use & reasoning |
| Writer | Mixtral-8x7B-Instruct | 47B | Long-form writing |
| Editor | Llama-3-8B-Instruct | 8B | Review & refinement |

All models use 4-bit quantization for efficiency.

## Deployment

### Docker

```bash
# Build and run
docker-compose up --build

# Production with PostgreSQL
docker run -d \
  -p 8000:8000 \
  -e DATABASE_URL="postgresql://user:pass@host/db" \
  ai-research-agent:latest
```

### Hugging Face Spaces

1. Create new Space with Docker SDK
2. Push code with HF-compatible `README.md`:
   ```yaml
   ---
   title: AI Research Agent
   sdk: docker
   app_port: 8000
   ---
   ```
3. Add `HF_TOKEN` as repository secret

## Performance

**Hardware Requirements**:
- Minimum: 16GB RAM (CPU-only with quantization)
- Recommended: 16GB RAM + 16GB VRAM GPU
- Optimal: 24GB+ VRAM GPU

**Typical Workflow**:
- Research query → 2-3 minutes
- Generated report: 1500-3000 words with citations

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open a Pull Request

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for Transformers
- [Mistral AI](https://mistral.ai/), [Meta AI](https://ai.meta.com/), [Nous Research](https://nousresearch.com/) for models
- [DuckDuckGo](https://duckduckgo.com/), [arXiv](https://arxiv.org/), [Wikipedia](https://wikipedia.org/) for research tools

## Contact

- **Issues**: [GitHub Issues](https://github.com/Kwesisbits/ai-research-agent/issues)
- **Email**: your.email@example.com

---

