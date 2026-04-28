import os
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import OpenAI

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_RESP")
api_key = os.getenv("AZURE_OPENAI_API_KEY_RESP")

if not endpoint:
    raise RuntimeError("Falta la variable AZURE_OPENAI_ENDPOINT_RESP")

# Normaliza endpoint por si viene con / final
base_url = endpoint.rstrip("/") + "/openai/v1/"
'''
# Para Responses API v1, usa scope de Azure AI
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://ai.azure.com/.default"
)
'''

client = OpenAI(
    base_url=base_url,
    api_key=api_key,
)

DEPLOYMENT_NAME = "gpt-4o"  

response = client.responses.create(
    model=DEPLOYMENT_NAME,
    instructions="Eres un asistente técnico conciso.",
    input="Explica en dos frases qué es Azure OpenAI dentro de una arquitectura de agentes.",
    temperature=0.2,
    max_output_tokens=200,
)

print(response.output_text)