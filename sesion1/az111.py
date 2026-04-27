import os
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-10-21",
    azure_ad_token_provider=token_provider,
)

response = client.chat.completions.create(
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_1", "gpt-5-reasoning-prod"),
    messages=[
        {
            "role": "developer",
            "content": (
                "Eres un arquitecto de IA enterprise. "
                "Responde en español, de forma concisa y accionable."
            )
        },
        {
            "role": "user",
            "content": (
                "Diseña una estrategia de escalado para un agente de soporte "
                "con varias tools."
            )
        }
    ],
    max_completion_tokens=500
)

print(response.choices[0].message.content)