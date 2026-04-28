import os
from typing import List

from dotenv import load_dotenv

from build_client import build_client_chat


# Carga variables desde .env
load_dotenv("../.env")

API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

PRIMARY = {
    "name": "primary",
    "endpoint": os.getenv("AOAI_ENDPOINT_PRIMARY"),
    "api_key": os.getenv("AOAI_API_KEY_PRIMARY"),
    "deployment": os.getenv("AOAI_DEPLOYMENT_CHAT_PRIMARY", "gpt-4o"),
}

SECONDARY = {
    "name": "secondary",
    "endpoint": os.getenv("AOAI_ENDPOINT_SECONDARY"),
    "api_key": os.getenv("AOAI_API_KEY_SECONDARY"),
    "deployment": os.getenv("AOAI_DEPLOYMENT_CHAT_SECONDARY", "gpt-4o"),
}

def safe_chat(prompt: str) -> str:
    """
    Intenta ejecutar el chat contra el endpoint primario.
    Si falla, prueba con el secundario.

    La autenticación queda delegada en build_client_chat:
    - si AZURE_OPENAI_API_KEY existe, usa API Key
    - si no existe o está vacía, usa RBAC / Microsoft Entra ID
    """

    for config in [PRIMARY, SECONDARY]:
        endpoint = config["endpoint"]
        api_key = config["api_key"]
        deployment = config["deployment"]
    
        try:
            client = build_client_chat(
                endpoint=endpoint,
                api_key=api_key,
                api_version=API_VERSION,
            )

            response = client.chat.completions.create(
                model=deployment,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Eres un asistente técnico. "
                            "Responde en español de forma clara y concisa."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.2,
            )

            content = response.choices[0].message.content

            if not content:
                raise RuntimeError(
                    f"El endpoint {endpoint} respondió sin contenido."
                )

            return content

        except Exception as ex:
            print(f"Fallo en endpoint {endpoint}: {ex}")


if __name__ == "__main__":
    print(safe_chat("Resume qué aporta un endpoint secundario."))