import logging

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from src.config import (
    CHAT_MODEL,
    FLASK_HOST,
    FLASK_PORT,
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    RETRIEVAL_TOP_K,
)
from src.helper import build_embedding_model
from src.prompt import SYSTEM_PROMPT

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mediassist")

app = Flask(__name__)


def validate_env() -> None:
    missing = []
    if not PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}"
        )


validate_env()

embeddings = build_embedding_model()
vector_store = PineconeVectorStore.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings,
)
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": RETRIEVAL_TOP_K},
)

chat_model = ChatOpenAI(model=CHAT_MODEL, temperature=0.2)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ]
)

qa_chain = create_stuff_documents_chain(chat_model, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)


@app.route("/")
def home():
    return render_template("chat.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "MediAssist-AI"})


@app.route("/get", methods=["POST"])
def chat():
    user_message = request.form.get("msg", "").strip()
    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400

    try:
        response = rag_chain.invoke({"input": user_message})
        answer = response.get("answer", "Unable to generate a response.")
        logger.info("Query processed successfully.")
        return answer
    except Exception as exc:
        logger.exception("Failed to process chat request.")
        return jsonify({"error": "Something went wrong. Please try again."}), 500


if __name__ == "__main__":
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=True)
