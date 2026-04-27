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

EMBEDDING_DEPLOYMENT = "text-embedding-3-small"

texts = [
    "Azure OpenAI proporciona inferencia sobre modelos en Azure.",
    "LangGraph ayuda a orquestar flujos con estado.",
    "Azure AI Search permite recuperación sobre documentos indexados."
]

response = client.embeddings.create(
    model=EMBEDDING_DEPLOYMENT,
    input=texts
)

print(f"Embeddings generados: {len(response.data)}")
print(f"Dimensión del primer embedding: {len(response.data[0].embedding)}")