import os
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import MemorySearchPreviewTool, PromptAgentDefinition


project_client = AIProjectClient(
    endpoint=os.environ["FOUNDRY_PROJECT"],
    credential=DefaultAzureCredential(exclude_interactive_browser_credential=False),
)

memory_store_name = os.environ["FOUNDRY_MEMORY_STORE_NAME"]

memory_tool = MemorySearchPreviewTool(
    memory_store_name=memory_store_name,
    scope="{{$userId}}",
    update_delay=1,  
)

agent = project_client.agents.create_version(
    agent_name="SupportAgentWithMemoryScope",
    definition=PromptAgentDefinition(
        model=os.environ["AOAI_DEPLOYMENT_CHAT_PRIMARY"],
        instructions=(
            "Eres un agente de soporte técnico. "
            "Usa la memoria solo para preferencias del usuario y continuidad útil. "
            "No uses memoria como fuente documental; para políticas o procedimientos usa RAG."
        ),
        tools=[memory_tool],
    ),
)

print(f"Agente creado: {agent.name} v{agent.version}")