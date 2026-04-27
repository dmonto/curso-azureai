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

DEPLOYMENT_NAME = "gpt-4o"  

response = client.chat.completions.create(
    model=DEPLOYMENT_NAME,
    messages=[
        {"role": "system", "content": "Eres un asistente técnico conciso."},
        {"role": "user", "content": "Explica en dos frases qué es Azure OpenAI dentro de una arquitectura de agentes."}
    ],
    temperature=0.2,
    max_tokens=200
)

print(response.choices[0].message.content)