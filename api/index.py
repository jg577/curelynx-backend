import logging
import json
from flask import Flask, request
from flask_cors import CORS, cross_origin
import os
from openai import OpenAI
import requests
import json

app = Flask(__name__)
app.logger.setLevel(logging.INFO)
CORS(app, origins=["*"])

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_ORGANIZATION = os.environ["OPENAI_ORGANIZATION"]
OPENAI_EMBEDDING_MODEL_NAME = os.environ["OPENAI_EMBEDDING_MODEL_NAME"]
AWS_ACCESS_KEY_IAM = os.environ["AWS_ACCESS_KEY_IAM"]
AWS_SECRET_KEY_IAM = os.environ["AWS_SECRET_KEY_IAM"]
AWS_OPENSEARCH_MASTER = os.environ["AWS_OPENSEARCH_MASTER"]
AWS_OPENSEARCH_PASS = os.environ["AWS_OPENSEARCH_PASS"]
AWS_OPENSEARCH_REGION = os.environ["AWS_OPENSEARCH_REGION"]
AWS_OPENSEARCH_URI = os.environ["AWS_OPENSEARCH_URI"]
AWS_OPENSEARCH_INDEX = os.environ["AWS_OPENSEARCH_INDEX"]


openai_client = OpenAI(api_key=OPENAI_API_KEY)


@app.route("/", methods=["GET"])
@cross_origin()
def index():
    return "This is the flask app"


def get_opensearch_results(content):
    # AWS OpenSearch endpoint URL
    print(content)
    url = AWS_OPENSEARCH_URI + "/curelynx-dev-v2/_search"

    # Authentication credentials
    auth = (AWS_OPENSEARCH_MASTER, AWS_OPENSEARCH_PASS)

    # Elasticsearch query
    query = {
        "size": 5,
        "query": {
            "bool": {
                "should": [
                    {"match": {"condition": content["condition"]}},
                    {"match": {"condition_mesh_term": content["condition"]}},
                    {"match": {"brief_summary": content["condition"]}},
                ],
                "minimum_should_match": 1,
                "boost": 2,
                "filter": [
                    {"range": {"minimum_age": {"lte": int(content["age"])}}},
                    {"range": {"maximum_age": {"gte": int(content["age"])}}},
                    {
                        "bool": {
                            "should": [
                                {"match": {"gender": content["gender"]}},
                                {"match": {"gender": "All"}},
                            ]
                        }
                    },
                    {"bool": {"should": [{"match": {"cities": content["city"]}}]}},
                ],
            }
        },
    }

    # Convert the query to a JSON string
    query_json = json.dumps(query)

    # Send the request
    response = requests.get(
        url, auth=auth, data=query_json, headers={"Content-Type": "application/json"}
    )

    # Print the response
    data = response.json()
    results = []

    for hit in data["hits"]["hits"]:
        source = hit["_source"]
        result = {
            key: value
            for key, value in source.items()
            if key not in ["cities", "states", "countries", "locations"]
        }
        # hack for now
        result = {"metadata": {"text": result}}
        results.append(result)

    return results


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
                "content": "please read the patient note and infer the age, gender, disease/condition, city, state and country of the user and return just a json of these fields. Use MeSH term for condition name. Use int for age. The input might also have zipcode. If so use that zipcode to look up city, state and country. Use United States in country name instead of US or USA and camel case. For example {'age': '', gender:'','condition': '','city':'', 'state':'', 'country:''}",
            },
            {"role": "user", "content": query_text},
        ],
    )
    app.logger.info("Chat response is %s", chat_response)
    content = json.loads(chat_response.choices[0].message.content)
    app.logger.info("input dict is %s", content)

    combined_matches = get_opensearch_results(content)
    app.logger.info(" output is %s", combined_matches)
    return {"matches": combined_matches}


if __name__ == "__main__":
    app.run()
