import os

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    MemoryStoreDefaultDefinition,
    MemoryStoreDefaultOptions,
)
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential

endpoint = os.environ["AZURE_AI_PROJECT_ENDPOINT"]
credential = DefaultAzureCredential()
client = AIProjectClient(endpoint=endpoint, credential=credential)

store_name = "customer-memory-store"

try:
    store = client.beta.memory_stores.get(store_name)
    print(f"✓ Memory store '{store_name}' ya existe")
except ResourceNotFoundError:
    definition = MemoryStoreDefaultDefinition(
        chat_model="gpt-4o",
        embedding_model="text-embedding-3-small",
        options=MemoryStoreDefaultOptions(
            user_profile_enabled=True,
            chat_summary_enabled=True,
        ),
    )

    store = client.beta.memory_stores.create(
        name=store_name,
        description="Long-term memory store",
        definition=definition,
    )
    print(f"✓ Memory store '{store_name}' creado correctamente")