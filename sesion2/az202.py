import os
from pathlib import Path

from dotenv import load_dotenv

from build_client import build_client_resp


load_dotenv("../.env")


endpoint = os.getenv("AOAI_ENDPOINT_RESP") or os.getenv("AOAI_ENDPOINT_SECONDARY")
api_key = os.getenv("AOAI_API_KEY_RESP") or os.getenv("AOAI_API_KEY_SECONDARY")

if not endpoint:
    raise RuntimeError(
        "Falta la variable de entorno AZURE_OPENAI_ENDPOINT_RESP "
        "o AZURE_OPENAI_ENDPOINT"
    )


client = build_client_resp(
    endpoint=endpoint,
    api_key=api_key,
)


deployment_name = os.getenv(
    "AZURE_OPENAI_DEPLOYMENT_RESP",
    os.getenv("AZURE_OPENAI_DEPLOYMENT_1", "gpt-5.1-chat"),
)


response = client.responses.create(
    model=deployment_name, 

    instructions=(
        "Eres un arquitecto de IA enterprise. "
        "Responde en español, de forma concisa y accionable."
    ),

    input=(
        "Diseña una estrategia de escalado para un agente de soporte "
        "con varias tools."
    ),

    reasoning={
        "effort": "medium",
        "summary": "auto",
    },

    text={
        "verbosity": "medium",
    },

    max_output_tokens=500,
)

print(response.output_text)