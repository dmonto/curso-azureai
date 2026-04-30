import os

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    MemoryStoreDefaultDefinition,
    MemoryStoreDefaultOptions,
)


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Falta la variable de entorno: {name}")
    return value


project_client = AIProjectClient(
    endpoint=require_env("FOUNDRY_PROJECT"),
    credential=DefaultAzureCredential(exclude_interactive_browser_credential=False),
)

stores = list(project_client.beta.memory_stores.list())

for store in stores:
    print("NAME:", store.name)
    print("ID:", getattr(store, "id", None))
    print("DESCRIPTION:", getattr(store, "description", None))
    print("DEFINITION:", getattr(store, "definition", None))
    print("-" * 80)
    
memory_store_name = "support_agent_memory"

options = MemoryStoreDefaultOptions(
    chat_summary_enabled=True,
    user_profile_enabled=True,
    user_profile_details=(
        "Conservar solo preferencias útiles y contexto operativo. "
        "Evitar datos sensibles como credenciales, datos financieros, "
        "ubicación precisa, edad o información personal innecesaria."
    ),
)

definition = MemoryStoreDefaultDefinition(
    chat_model=require_env("AOAI_DEPLOYMENT_CHAT_PRIMARY"),
    embedding_model=require_env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    options=options,
)

memory_store = project_client.beta.memory_stores.create(
    name=memory_store_name,
    definition=definition,
    description="Memoria de largo plazo para agente de soporte del curso",
)

print("Memory store creado:")
print("Name:", memory_store.name)
print("Description:", memory_store.description)