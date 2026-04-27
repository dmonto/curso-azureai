import os

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient


PROJECT_ENDPOINT = os.getenv(
    "AZURE_AI_PROJECT_ENDPOINT",
    "https://curso-ia-openai.services.ai.azure.com/api/projects/curso-ia-openai-project"
)

DEPLOYMENT_NAME = os.getenv(
    "AZURE_OPENAI_DEPLOYMENT_1",
    "gpt-4o"
)


project = AIProjectClient(
    endpoint=PROJECT_ENDPOINT,
    credential=DefaultAzureCredential(exclude_interactive_browser_credential=False)
)

openai_client = project.get_openai_client()

print("Cliente OpenAI obtenido desde AIProjectClient")
print("Base URL:", openai_client.base_url)
print("Deployment:", DEPLOYMENT_NAME)
print()

response = openai_client.chat.completions.create(
    model=DEPLOYMENT_NAME,
    messages=[
        {
            "role": "system",
            "content": "Eres un asistente técnico conciso."
        },
        {
            "role": "user",
            "content": "Explica en una frase para qué sirve un project en Foundry."
        }
    ],
    temperature=0.2,
    max_tokens=120,
)

print(response.choices[0].message.content)