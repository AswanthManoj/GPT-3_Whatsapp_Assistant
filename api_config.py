import os


openai_key = "<your openai api key>"
cohere_key = "<your cohere api key>"
serpapi_key = "<your serpapi api key>"
ultramsg_token = "<your ultramsg access token>"
ultramsg_instance_id = "<your ultramsg instance id>"

os.environ["OPENAI_API_KEY"] = openai_key
os.environ["serpapi_api_key"] = serpapi_key
os.environ["COHERE_API_KEY"] = cohere_key
