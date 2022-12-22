from flask import Flask, request, jsonify
import Whatsapp_assistant
from api_config import openai_key, cohere_key, serpapi_key, ultramsg_token, ultramsg_instance_id
import json
from pprint import pprint


id = "918086639482@c.us"
ai = Whatsapp_assistant.assistant(ph_id=id, ultramsg_token=ultramsg_token, ultramsg_instance=ultramsg_instance_id)
ai.authenticate(openai_key=openai_key, cohere_key=cohere_key, serpapi_key=serpapi_key)
app = Flask(__name__)


@app.route('/', methods=['POST'])


async def home():

    if request.method == 'POST':
        state, data = ai.request_process(request.json)
        if state:
            ai.process(data["sender id"], data["message"])

    return ''


if(__name__) == '__main__':
    app.run(debug=True)