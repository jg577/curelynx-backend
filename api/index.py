import json
import logging
from flask import Flask, request
from flask_cors import CORS, cross_origin
from langchain import OpenAI
import requests
import os
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone

app = Flask(__name__)
app.logger.setLevel(logging.INFO)
CORS(app, origins=["*"])

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENVIRONMENT = os.environ["PINECONE_ENVIRONMENT"]
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_ORGANIZATION = os.environ["OPENAI_ORGANIZATION"]
OPENAI_EMBEDDING_MODEL_NAME = os.environ["OPENAI_EMBEDDING_MODEL_NAME"]


openai_emb_service = OpenAIEmbeddings(
    model=OPENAI_EMBEDDING_MODEL_NAME,
    openai_api_key=OPENAI_API_KEY,
    openai_organization=OPENAI_ORGANIZATION,
)
pinecone.init(api_key=PINECONE_API_KEY, environment="asia-southeast1-gcp-free")
index = pinecone.Index(PINECONE_INDEX_NAME)
vectorstore = Pinecone(index, openai_emb_service.embed_query, "text")

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(temperature=0), retriever=vectorstore.as_retriever(), memory=memory
)


@app.route("/", methods=["GET"])
@cross_origin()
def index():
    return "This is the flask app"


@app.route("/api/get_trials", methods=["POST"])
@cross_origin()
def get_trials():
    """
    This function gets the request and returns a list of clinical trials that could be useful
    """
    app.logger.info("Reached the function")
    data = request.get_json()
    # condition = data["condition"]
    query_text = data["question"]
    app.logger.info(f"Retrieved data into the function {data}")
    question_embedding = openai_emb_service.embed_query(query_text)
    app.logger.info(f"Got embedding information from openai: { question_embedding}")
    result = index.query(
        vector=question_embedding,
        top_k=10,
        include_metadata=True,
    ).to_dict()
    app.logger.info(f"Got results from the index: {result}")
    return result


@app.route("/api/start_conversation", methods=["POST"])
@cross_origin()
def start_conversation():
    """
    This function reads the question field from the request.
    It then creates a langchain agent with a llm that uses the question as a prompt and returns a response by also reading through a pinecone document.
    """
    query = request.get_json()["question"]
    result = qa({"question": query})
    return result["answer"]


if __name__ == "__main__":
    app.run()
