import asyncio
import os
from typing import cast

from dotenv import load_dotenv
from agent_framework import Agent, Message, AgentResponseUpdate
from agent_framework.orchestrations import GroupChatBuilder, GroupChatState

from build_client import build_maf_responses_client


load_dotenv("../.env")


def select_next_speaker(state: GroupChatState) -> str | None:
    sequence = [
        "AnalystAgent",
        "ReviewerAgent",
        "AnalystAgent",
        "FinalizerAgent",
    ]

    if state.current_round >= len(sequence) * 2:
        return None

    return sequence[state.current_round % len(sequence)] 


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
            "Identifica puntos clave, riesgos y decisiones necesarias. "
            "Si ya existe una revisión previa, incorpora sus mejoras."
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
            "Convierte la conversación del grupo en una respuesta final breve, clara y accionable."
        ),
    )

    workflow = GroupChatBuilder(
        participants=[
            analyst,
            reviewer,
            finalizer,
        ],
        selection_func=select_next_speaker,
        termination_condition=lambda conversation: len(conversation) >= 8,
        intermediate_outputs=True,
    ).build()

    outputs: list[list[Message]] = []
    last_response_id: str | None = None

    async for event in workflow.run(
        "Diseña una estrategia de escalado para un agente de soporte con varias tools.",
        stream=True,
    ):
        if event.type != "output":
            print(event.type)

        data = event.data

        if isinstance(data, AgentResponseUpdate):
            response_id = data.response_id

            if response_id != last_response_id:
                if last_response_id is not None:
                    print("\n")

                print(f"\n--- {data.author_name} ---")
                last_response_id = response_id

            print(data.text, end="", flush=True)

        elif isinstance(data, list):
            print("\n\n===== Conversación parcial =====")

            for i, message in enumerate(data, start=1):
                author = getattr(message, "author_name", None) or getattr(message, "role", "unknown")
                text = getattr(message, "text", "")

                print(f"\n--- Mensaje {i} | {author} ---")
                print(text)
                outputs.append(text)

    if not outputs:
        raise RuntimeError("El GroupChat no produjo salida.")

    print("===== Conversación final =====")
    for i, message in enumerate(outputs, start=1):
        role = getattr(message, "role", "unknown")
        content = getattr(message, "text", None) or getattr(message, "content", "")

        print(f"\n--- Mensaje {i} | {role} ---")
        print(content)


if __name__ == "__main__":
    asyncio.run(main())