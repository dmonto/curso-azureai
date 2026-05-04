import os
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import MemorySearchOptions


PROJECT_ENDPOINT = os.environ["FOUNDRY_PROJECT"]
MEMORY_STORE_NAME = os.environ["FOUNDRY_MEMORY_STORE_NAME"]


project_client = AIProjectClient(
    endpoint=PROJECT_ENDPOINT,
    credential=DefaultAzureCredential(exclude_interactive_browser_credential=False),
)


def search_static_memories(user_id: str):
    """
    Recupera memorias estáticas de perfil.
    Patrón: scope sin items.
    """
    response = project_client.beta.memory_stores.search_memories(
        name=MEMORY_STORE_NAME,
        scope=user_id,
        options=MemorySearchOptions(max_memories=5),
    )

    return [
        memory.memory_item.content
        for memory in response.memories
    ]


def search_contextual_memories(user_id: str, latest_user_message: str):
    """
    Recupera memorias relevantes al mensaje actual.
    Patrón: scope + items.
    """
    query_message = {
        "type": "message",
        "role": "user",
        "content": latest_user_message,
    }

    response = project_client.beta.memory_stores.search_memories(
        name=MEMORY_STORE_NAME,
        scope=user_id,
        items=[query_message],
        options=MemorySearchOptions(max_memories=5),
    )

    return [
        memory.memory_item.content
        for memory in response.memories
    ]


if __name__ == "__main__":
    user_id = "user_123"
    message = "Explícame cómo usar RAG con Azure AI Search para soporte técnico."

    static_memories = search_static_memories(user_id)
    contextual_memories = search_contextual_memories(user_id, message)

    print("=== MEMORIAS ESTÁTICAS ===")
    for memory in static_memories:
        print("-", memory)

    print("\n=== MEMORIAS CONTEXTUALES ===")
    for memory in contextual_memories:
        print("-", memory)