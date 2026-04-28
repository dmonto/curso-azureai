import asyncio
from typing import Annotated
import os

from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatCompletionClient
from pydantic import Field
from azure.identity.aio import AzureCliCredential


@tool(approval_mode="never_require")
def get_weather(
    location: Annotated[str, Field(description="Ciudad de la que se quiere saber el tiempo")]
) -> str:
    return f"El tiempo en {location} es soleado y 24°C."


async def main() -> None:
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    #credential = AzureCliCredential()
    client = OpenAIChatCompletionClient(
        azure_endpoint="https://curso-ia-openai.openai.azure.com/",
        model="gpt-4o",             
        api_version="2024-10-21",
        api_key=api_key,
    )

    agent = Agent(
        client=client,
        name="WeatherAgent",
        instructions="Usa tools cuando sea útil y responde en castellano.",
        tools=[get_weather],
    )

    result = await agent.run("¿Qué tiempo hace en Madrid?")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())