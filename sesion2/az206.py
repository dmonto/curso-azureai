import asyncio
import os
from pathlib import Path
from typing import cast

from dotenv import load_dotenv
from agent_framework import Agent, Message
from agent_framework.orchestrations import SequentialBuilder

from build_client import build_maf_responses_client


load_dotenv("../.env")


async def main() -> None:
    client = build_maf_responses_client(
        endpoint=os.getenv("AOAI_ENDPOINT_SECONDARY"),
        model=os.getenv("AOAI_DEPLOYMENT_CHAT_SECONDARY", "gpt-5.1-chat"),
        api_key=os.getenv("AOAI_API_KEY_SECONDARY"),
        api_version=os.getenv("AZURE_OPENAI_RESPONSES_API_VERSION", "preview"),
    )

    analyst = Agent(
        client=client,
        name="AnalystAgent",
        instructions=(
            "Eres un analista técnico. "
            "Identifica los puntos clave, riesgos y decisiones necesarias."
        ),
    )

    reviewer = Agent(
        client=client,
        name="ReviewerAgent",
        instructions=(
            "Eres un revisor senior. "
            "Revisa la respuesta anterior, corrige imprecisiones y añade mejoras."
        ),
    )

    finalizer = Agent(
        client=client,
        name="FinalizerAgent",
        instructions=(
            "Eres un sintetizador ejecutivo. "
            "Convierte el análisis y la revisión en una respuesta final breve, clara y accionable."
        ),
    )

    workflow = SequentialBuilder(
        participants=[
            analyst,
            reviewer,
            finalizer,
        ]
    ).build()

    outputs: list[list[Message]] = []

    async for event in workflow.run(
        "Diseña una estrategia de escalado para un agente de soporte con varias tools.",
        stream=True,
    ):
        if event.type == "output":
            outputs.append(cast(list[Message], event.data))

    if not outputs:
        raise RuntimeError("El workflow no produjo salida.")

    final_conversation = outputs[-1]

    print("===== Conversación final =====")
    for i, message in enumerate(final_conversation, start=1):
        role = getattr(message, "role", "unknown")
        content = getattr(message, "text", None) or getattr(message, "content", "")

        print(f"\n--- Mensaje {i} | {role} ---")
        print(content)


if __name__ == "__main__":
    asyncio.run(main())