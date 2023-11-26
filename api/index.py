import logging
import json
from flask import Flask, request
from flask_cors import CORS, cross_origin
import os
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from openai import OpenAI


app = Flask(__name__)
app.logger.setLevel(logging.INFO)
CORS(app, origins=["*"])

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENVIRONMENT = os.environ["PINECONE_ENVIRONMENT"]
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_ORGANIZATION = os.environ["OPENAI_ORGANIZATION"]
OPENAI_EMBEDDING_MODEL_NAME = os.environ["OPENAI_EMBEDDING_MODEL_NAME"]

openai_client = OpenAI(api_key=OPENAI_API_KEY)
openai_emb_service = OpenAIEmbeddings(
    model=OPENAI_EMBEDDING_MODEL_NAME,
    openai_api_key=OPENAI_API_KEY,
    openai_organization=OPENAI_ORGANIZATION,
)
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
pinecone_index = pinecone.Index(PINECONE_INDEX_NAME)


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
    # get parsed_location for metadata filter:
    app.logger.info("Query text is %s", query_text)
    chat_response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "please read the patient note and infer the city, state and country of the user and return just a json of these three fields. Use United States in country name instead of US or USA. For example {'city':'', 'state':'', 'country:''}",
            },
            {"role": "user", "content": query_text},
        ],
    )
    app.logger.info("Chat response is %s", chat_response)
    location_dict = json.loads(chat_response.choices[0].message.content)
    app.logger.info("location dict is %s", location_dict)
    question_embedding = openai_emb_service.embed_query(query_text)
    k = 5
    result_city = pinecone_index.query(
        vector=question_embedding,
        filter={
            "city": {"$eq": location_dict["city"]},
        },
        top_k=k,
        include_metadata=True,
    ).to_dict()
    result_state = pinecone_index.query(
        vector=question_embedding,
        filter={
            "state": {"$eq": location_dict["state"]},
        },
        top_k=k,
        include_metadata=True,
    ).to_dict()
    result_country = pinecone_index.query(
        vector=question_embedding,
        filter={
            "country": {"$eq": location_dict["country"]},
        },
        top_k=k,
        include_metadata=True,
    ).to_dict()
    results_no_filter = pinecone_index.query(
        vector=question_embedding,
        top_k=k,
        include_metadata=True,
    ).to_dict()

    # combinining matches
    n_matches = k
    trial_ids = []
    matches = []
    while n_matches >= 0:
        combined_results = {
            **result_city,
            **result_state,
            **result_country,
            **results_no_filter,
        }
        i = 0
        if combined_results.matches[i].metadata.NCTId not in trial_ids:
            trial_ids.append(combined_results.matches[0].metadata.NCTId)
            matches.append(combined_results.matches[i])
            i += 1
            n_matches += -1

    app.logger.info("Got results from the index: %s", combined_results)
    return combined_results


if __name__ == "__main__":
    app.run()
