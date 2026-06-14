# MediAssist-AI

**MediAssist-AI** is a production-ready Retrieval-Augmented Generation (RAG) chatbot for medical question answering. It combines LangChain, OpenAI, Pinecone vector search, and a Flask web interface to deliver context-aware responses from your medical knowledge base.

Built and maintained by **[Shraddha3838](https://github.com/Shraddha3838)** (Coder Shraddha).

---

## Features

- PDF ingestion pipeline with chunking and metadata normalization
- HuggingFace embeddings (`all-MiniLM-L6-v2`) stored in Pinecone
- GPT-powered RAG responses with safety-focused system prompts
- Responsive chat UI with real-time AJAX messaging
- Docker support and GitHub Actions CI/CD for AWS ECR + EC2 deployment
- Health check endpoint at `/health`

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Python, Flask |
| Orchestration | LangChain |
| LLM | OpenAI GPT |
| Vector DB | Pinecone |
| Embeddings | Sentence Transformers |
| Deployment | Docker, AWS ECR, EC2, GitHub Actions |

---

## Project Structure

```text
mediassist-ai/
├── app.py                 # Flask application entrypoint
├── store_index.py         # Embedding upload script
├── src/
│   ├── config.py          # Environment-driven configuration
│   ├── helper.py          # PDF loading and text splitting utilities
│   └── prompt.py          # RAG system prompt
├── data/                  # Place medical PDF files here
├── templates/             # HTML templates
├── static/                # CSS assets
├── Dockerfile
└── .github/workflows/     # CI/CD pipeline
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Shraddha3838/git.git
cd git
```

### 2. Create a virtual environment

```bash
conda create -n mediassist python=3.10 -y
conda activate mediassist
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example file and add your API keys:

```bash
cp .env.example .env
```

```ini
PINECONE_API_KEY=your_pinecone_key
OPENAI_API_KEY=your_openai_key
```

### 5. Add medical documents

Place your PDF files inside the `data/` directory.

### 6. Build the vector index

```bash
python store_index.py
```

### 7. Run the application

```bash
python app.py
```

Open **http://localhost:8080** in your browser.

---

## Docker

```bash
docker build -t mediassist-ai .
docker run -p 8080:8080 --env-file .env mediassist-ai
```

---

## AWS Deployment

1. Create an ECR repository and EC2 instance with Docker installed.
2. Configure a self-hosted GitHub Actions runner on EC2.
3. Add these repository secrets:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `AWS_DEFAULT_REGION`
   - `ECR_REPO`
   - `PINECONE_API_KEY`
   - `OPENAI_API_KEY`

Push to `main` to trigger automated build and deployment.

---

## Disclaimer

MediAssist-AI is intended for educational and informational use only. It does not provide medical diagnosis or emergency guidance. Always consult a licensed healthcare provider for medical decisions.

---

## Author

**Shraddha3838** — Fullstack Developer & AIML Engineer  
GitHub: [https://github.com/Shraddha3838](https://github.com/Shraddha3838)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
