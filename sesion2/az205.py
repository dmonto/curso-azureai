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
        instructions="Recuerda el contexto de esta conversación.",
    )

    session = agent.create_session()

    nombre = input("Como te llamas? ")
    r1 = await agent.run(f"Mi nombre es {nombre}.", session=session)
    print(r1)

    r2 = await agent.run("¿Cómo me llamo?", session=session)
    print(r2)



if __name__ == "__main__":
    asyncio.run(main())