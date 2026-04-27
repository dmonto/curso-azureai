import os

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import OpenAI


endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_RESP")
if not endpoint:
    raise RuntimeError("Falta la variable de entorno AZURE_OPENAI_ENDPOINT_RESP")

base_url = endpoint.rstrip("/") + "/openai/v1/"

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://ai.azure.com/.default"
)

client = OpenAI(
    base_url=base_url,
    api_key=token_provider,
)

deployment_name = os.getenv(
    "AZURE_OPENAI_DEPLOYMENT_1",
    "gpt-5.1-chat"
)

response = client.responses.create(
    model=deployment_name,  # nombre del deployment en Azure

    instructions=(
        "Eres un arquitecto de IA enterprise. "
        "Responde en español, de forma concisa y accionable."
    ),

    input=(
        "Diseña una estrategia de escalado para un agente de soporte "
        "con varias tools."
    ),

    reasoning={
        "effort": "medium",   # low, medium, high; según modelo también none/minimal/xhigh
        "summary": "auto"     # auto, concise o detailed; en GPT-5 no usar concise
    },

    text={
        "verbosity": "medium"    # low, medium, high
    },

    max_output_tokens=500,
)

print(response.output_text)