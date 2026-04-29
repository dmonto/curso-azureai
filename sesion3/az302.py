import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from agent_framework import Agent

try:
    from agent_framework.orchestrations import HandoffBuilder
except ImportError:
    from agent_framework import HandoffBuilder

from build_client import build_maf_responses_client


load_dotenv("../.env")

def should_terminate(messages: list) -> bool:
    if len(messages) >= 8:
        return True
    
async def main() -> None:
    client = build_maf_responses_client(
        endpoint=os.getenv("AOAI_ENDPOINT_SECONDARY"),
        model=os.getenv("AOAI_DEPLOYMENT_CHAT_SECONDARY", "gpt-5.1-chat"),
        api_key=os.getenv("AOAI_API_KEY_SECONDARY"),
        api_version=os.getenv("AZURE_OPENAI_RESPONSES_API_VERSION", "preview"),
    )


    triage_agent = Agent(
        client=client,
        name="triage",
        description="Agente de triaje que decide qué especialista debe atender la petición.",
        require_per_service_call_history_persistence=True,
        instructions=(
            "Eres un agente de triaje. "
            "Si la petición trata de errores técnicos, VPN, acceso o incidencias, "
            "haz handoff al agente 'support'. "
            "Si trata de facturas, cobros o pagos, haz handoff al agente 'billing'. "
            "No resuelvas tú la petición si hay un especialista adecuado."
        ),
    )

    support_agent = Agent(
        client=client,
        name="support",
        description="Especialista en soporte técnico, accesos, VPN e incidencias.",
        require_per_service_call_history_persistence=True,
        instructions=(
            "Eres un agente de soporte técnico. "
            "Diagnostica la incidencia, pide datos mínimos si faltan y propone el siguiente paso."
        ),
    )

    billing_agent = Agent(
        client=client,
        name="billing",
        description="Especialista en facturación, pagos y cobros.",
        require_per_service_call_history_persistence=True,
        instructions=(
            "Eres un agente de facturación. "
            "Ayuda con dudas de facturas, cobros, pagos y datos administrativos."
        ),
    )

    workflow = (
        HandoffBuilder(
            name="support_handoff_workflow",
            participants=[
                triage_agent,
                support_agent,
                billing_agent,
            ],
            description="Workflow de handoff entre triaje, soporte y facturación.",
            termination_condition=should_terminate
        )
        .add_handoff(triage_agent, [support_agent, billing_agent])
        .add_handoff(support_agent, [triage_agent])
        .add_handoff(billing_agent, [triage_agent])
        .with_start_agent(triage_agent)
        .build()
    )

    events = await workflow.run(
        "Tengo un error al conectarme por VPN desde casa."
    )
    for event in events:
        print("=" * 80)
        print("EVENT:", event.type)
        print(event.data)

if __name__ == "__main__":
    asyncio.run(main())