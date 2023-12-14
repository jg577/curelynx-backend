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
    condition_list = []
    chat_response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "please read the patient note and infer the disease/condition, city, state and country of the user and return just a json of these four fields. Use MeSH term for condition name.  Use United States in country name instead of US or USA and camel case. For example {'condition': '','city':'', 'state':'', 'country:''}",
            },
            {"role": "user", "content": query_text},
        ],
    )
    app.logger.info("Chat response is %s", chat_response)
    content = json.loads(chat_response.choices[0].message.content)
    app.logger.info("location dict is %s", content)
    question_embedding = openai_emb_service.embed_query(query_text)
    k = 5
    results_city = pinecone_index.query(
        vector=question_embedding,
        filter={
            "city": {"$eq": content["city"]},
            "condition": {"$eq": content["condition"]},
        },
        top_k=k,
        include_metadata=True,
    ).to_dict()
    app.logger.info(
        f"city_results: {[match['metadata']['NCTId'] for match in results_city['matches']]}"
    )
    results_state = pinecone_index.query(
        vector=question_embedding,
        filter={
            "state": {"$eq": content["state"]},
            "condition": {"$eq": content["condition"]},
        },
        top_k=k,
        include_metadata=True,
    ).to_dict()
    app.logger.info(
        f"state_results: {[match['metadata']['NCTId'] for match in results_state['matches']]}"
    )
    results_country = pinecone_index.query(
        vector=question_embedding,
        filter={
            "country": {"$eq": content["country"]},
            "condition": {"$eq": content["condition"]},
        },
        top_k=k,
        include_metadata=True,
    ).to_dict()
    app.logger.info(
        f"county_results: {[match['metadata']['NCTId'] for match in results_country['matches']]}"
    )
    results_no_filter = pinecone_index.query(
        vector=question_embedding,
        top_k=k,
        filter={"condition": {"$eq": content["condition"]}},
        include_metadata=True,
    ).to_dict()
    app.logger.info(
        f"other_results: {[match['metadata']['NCTId'] for match in results_no_filter['matches']]}"
    )
    # combinining matches
    n_matches_left = k
    trial_ids = []
    combined_matches = []
    location_index = 0
    location_dict_list = [
        results_city,
        results_state,
        results_country,
        results_no_filter,
    ]
    while location_index < len(location_dict_list):
        location_dict = location_dict_list[location_index]
        app.logger.info(f"for index {location_index} location dict is {location_dict}")
        for match in location_dict["matches"]:
            if match["metadata"]["NCTId"] not in trial_ids:
                trial_ids.append(match["metadata"]["NCTId"])
                combined_matches.append(match)
                n_matches_left += -1
        location_index += 1

    # app.logger.info("Got results from the index: %s", combined_matches)
    return {"matches": combined_matches[:k]}


if __name__ == "__main__":
    app.run()
