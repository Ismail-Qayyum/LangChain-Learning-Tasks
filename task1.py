import langchain 
from dotenv import load_dotenv
import os



load_dotenv()
api_version = os.getenv("API_VERSION")
deployment_name = os.getenv("DEPLOYMENT_NAME")
endpoint_url = os.getenv("ENDPOINT_URL")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
print(f'version: {api_version}\n deployment_name: {deployment_name} \n endpoint url: {endpoint_url}\n api: {api_key}')
