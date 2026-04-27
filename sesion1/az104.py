import asyncio

from agent_framework import Agent
from agent_framework.openai import OpenAIChatCompletionClient
from azure.identity.aio import AzureCliCredential


async def main() -> None:
    credential = AzureCliCredential()
    try:
        client = OpenAIChatCompletionClient(
            azure_endpoint="https://curso-ia-openai.openai.azure.com/",
            model="gpt-4o",             
            api_version="2024-10-21",
            credential=credential,
        )

        agent = Agent(
            client=client,
            name="HelloAgent",
            instructions="Eres un asistente especialista en Azure AI. Responde de forma breve y clara.",
        )

        result = await agent.run("Explica en una frase el papel de Microsoft Agent Framework.")
        print(result)
    finally:
        await credential.close()


if __name__ == "__main__":
    asyncio.run(main())