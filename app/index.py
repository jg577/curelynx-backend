import json
import logging
from flask import Flask, request
from flask_cors import CORS, cross_origin
import requests
import os
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings


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


@app.route("/")
@app.route("/api/get_trials", methods=["POST"])
@cross_origin()
def get_trials():
    """
    This function gets the request and returns a list of clinical trials that could be useful
    """
    app.logger.info(request.data)
    data = json.load(request.data)
    # condition = data["condition"]
    query_text = data["question"]
    question_embedding = openai_emb_service.embed_query(query_text)
    pinecone.init(api_key=PINECONE_API_KEY, environment="asia-southeast1-gcp-free")
    index = pinecone.GRPCIndex(PINECONE_INDEX_NAME)
    result = index.query(
        vector=question_embedding,
        top_k=4,
        include_metadata=True,
    )
    print(result)
    response = {}
    return response
