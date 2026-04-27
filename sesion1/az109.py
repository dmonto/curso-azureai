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
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Clasifica la consulta en soporte, ventas o facturación. Devuelve solo la etiqueta."},
        {"role": "user", "content": "No puedo acceder a la VPN"}
    ],
    temperature=0,
    max_tokens=20,
    response_format={"type": "text"}
)

print(response.choices[0].message.content)