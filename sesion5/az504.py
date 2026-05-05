import os
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from dotenv import load_dotenv

load_dotenv("../.env")
endpoint="https://curso-ia-openai-responses.services.ai.azure.com/api/projects/responses-project"

user_id = os.environ["USER_ID"]

project_client = AIProjectClient(
    endpoint=endpoint,
    credential=DefaultAzureCredential(exclude_interactive_browser_credential=False),
)

agent = project_client.agents.get(agent_name="SupportAgentWithMemoryScope")

print("Agent:", agent.name)
print("Latest status:", agent.versions["latest"]["status"])

for v in project_client.agents.list_versions(agent_name="SupportAgentWithMemoryScope"):
    print("Version:", v.version, "Status:", v["status"])

with project_client.get_openai_client() as openai:
    conversation = openai.conversations.create()

    r1 = openai.responses.create(
        conversation=conversation.id,
        input="Mi nombre es Ana y prefiero respuestas breves.",
        extra_headers={
            "x-memory-user-id": user_id
        },        
        extra_body={
            "agent_reference": {
                "name": "SupportAgentWithMemoryScope",
                "type": "agent_reference",
            }
        },
    )

    print(r1.output_text)

    
    r2 = openai.responses.create(
        conversation=conversation.id,
        input="¿Qué sabes de mi?",
        extra_headers={
            "x-memory-user-id": user_id
        },        
        extra_body={
            "agent_reference": {
                "name": "SupportAgentWithMemoryScope",
                "type": "agent_reference",
            }
        },
    )

    print(r2.output_text)  

