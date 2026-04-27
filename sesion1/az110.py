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


schema = {
    "name": "ticket_triage",
    "schema": {
        "type": "object",
        "properties": {
            "category": {"type": "string"},
            "priority": {"type": "string"},
            "requires_human": {"type": "boolean"}
        },
        "required": ["category", "priority", "requires_human"],
        "additionalProperties": False
    }
}

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Analiza la incidencia y devuelve un objeto JSON válido."},
        {"role": "user", "content": "El cliente no puede facturar desde ayer y hay urgencia alta."}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": schema
    },
    temperature=0.2,
    max_tokens=120
)

print(response.choices[0].message.content)