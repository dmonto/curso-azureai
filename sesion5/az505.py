import os
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import MemorySearchOptions
from dotenv import load_dotenv

load_dotenv("../.env")

project_client = AIProjectClient(
    endpoint=os.environ["FOUNDRY_PROJECT"],
    credential=DefaultAzureCredential(),
)

memory_store_name = os.environ["FOUNDRY_MEMORY_STORE_NAME"]
scope = os.environ["USER_ID"]

search_response = project_client.beta.memory_stores.search_memories(
    name=memory_store_name,
    scope=scope,
    options=MemorySearchOptions(max_memories=20),
)

print(f"Memorias encontradas: {len(search_response.memories)}")

for memory in search_response.memories:
    print("-" * 80)
    print(memory.memory_item.content)    