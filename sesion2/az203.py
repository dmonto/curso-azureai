import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from agent_framework import Agent

from build_client import build_maf_responses_client

load_dotenv("../.env")


async def main() -> None:
    endpoint = os.getenv("AOAI_ENDPOINT_SECONDARY")
    api_key = os.getenv("AOAI_API_KEY_SECONDARY")
    api_version = os.getenv("AOAI_API_VERSION", "preview")
    deployment = os.getenv("AOAI_DEPLOYMENT_SECONDARY", "gpt-4o")

    client = build_maf_responses_client(
        endpoint=endpoint,
        model=deployment,
        api_key=api_key,
        api_version=api_version,
    )

    agent = Agent(
        client=client,
        name="SecondaryResponsesAgent",
        instructions=(
            "Eres un asistente especialista en Azure AI. "
            "Responde de forma breve y clara."
        ),
    )

    result = await agent.run(
        "Explica en una frase qué aporta usar Responses API en MAF."
    )

    print(result)



if __name__ == "__main__":
    asyncio.run(main())