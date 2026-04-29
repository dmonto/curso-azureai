import asyncio
import os
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv

from agent_framework import (
    Agent,
    AgentExecutorResponse,
    Executor,
    Message,
    WorkflowContext,
    WorkflowEvent,
    handler,
)

from agent_framework.orchestrations import SequentialBuilder

from build_client import build_maf_responses_client


load_dotenv("../.env")


@dataclass
class SupportDecision:
    category: str
    risk: Literal["bajo", "medio", "alto"]
    action: Literal["responder", "pedir_mas_info", "escalar"]
    reason: str


class RiskGateExecutor(Executor):
    """
    Executor de lógica custom.

    No usa LLM.
    Aplica reglas deterministas sobre la conversación generada por el agente anterior.
    """

    def __init__(self, id: str = "RiskGateExecutor") -> None:
        super().__init__(id=id)

    @handler
    async def evaluate(
        self,
        agent_response: AgentExecutorResponse,
        ctx: WorkflowContext[list[Message]],
    ) -> None:
        await ctx.add_event(
            WorkflowEvent(
                type="progress",
                data="RiskGateExecutor: evaluando riesgo de la incidencia",
            )
        )

        conversation = list(agent_response.full_conversation or [])

        full_text = "\n".join(
            getattr(message, "text", "") or ""
            for message in conversation
        ).lower()

        decision = self._decide(full_text)

        await ctx.add_event(
            WorkflowEvent(
                type="risk_decision",
                data={
                    "category": decision.category,
                    "risk": decision.risk,
                    "action": decision.action,
                    "reason": decision.reason,
                },
            )
        )

        decision_message = Message(
            role="assistant",
            contents=[
                (
                    "## Decisión operativa del workflow\n\n"
                    f"- Categoría: {decision.category}\n"
                    f"- Riesgo: {decision.risk}\n"
                    f"- Acción: {decision.action}\n"
                    f"- Motivo: {decision.reason}\n\n"
                    "Esta decisión ha sido calculada por lógica determinista del workflow, "
                    "no por el modelo."
                )
            ],
        )

        output_conversation = conversation + [decision_message]

        await ctx.send_message(output_conversation)

    def _decide(self, text: str) -> SupportDecision:
        category = "soporte_general"

        if "vpn" in text:
            category = "vpn"
        elif "factura" in text or "cobro" in text:
            category = "facturacion"
        elif "correo" in text or "email" in text:
            category = "correo"

        high_risk_terms = [
            "urgente",
            "producción",
            "produccion",
            "cliente",
            "caído",
            "caido",
            "bloqueado",
            "no puedo trabajar",
        ]

        missing_info_terms = [
            "no indica",
            "falta",
            "necesita más información",
            "necesita mas informacion",
            "no se especifica",
        ]

        has_high_risk = any(term in text for term in high_risk_terms)
        needs_more_info = any(term in text for term in missing_info_terms)

        if has_high_risk:
            return SupportDecision(
                category=category,
                risk="alto",
                action="escalar",
                reason=(
                    "La incidencia contiene señales de urgencia, impacto en cliente "
                    "o posible afectación operativa."
                ),
            )

        if needs_more_info:
            return SupportDecision(
                category=category,
                risk="medio",
                action="pedir_mas_info",
                reason=(
                    "El análisis detecta que falta información mínima para resolver "
                    "la incidencia con seguridad."
                ),
            )

        return SupportDecision(
            category=category,
            risk="bajo",
            action="responder",
            reason="No se detectan señales de alto riesgo ni falta crítica de información.",
        )


async def main() -> None:
    client = build_maf_responses_client(
        endpoint=os.getenv("AOAI_ENDPOINT_SECONDARY"),
        model=os.getenv("AOAI_DEPLOYMENT_CHAT_SECONDARY", "gpt-5.1-chat"),
        api_key=os.getenv("AOAI_API_KEY_SECONDARY"),
        api_version=os.getenv("AZURE_OPENAI_RESPONSES_API_VERSION", "preview"),
    )

    triage_agent = Agent(
        client=client,
        name="TriageAgent",
        instructions=(
            "Eres un agente de soporte nivel 1. "
            "Analiza la incidencia del usuario. "
            "Debes identificar categoría, posible causa, datos que faltan, impacto y siguiente acción. "
            "No crees tickets ni inventes IDs. "
            "Si falta información, indícalo claramente."
        ),
    )

    risk_gate = RiskGateExecutor()

    workflow = SequentialBuilder(
        participants=[
            triage_agent,
            risk_gate,
        ]
    ).build()

    user_input = (
        "No puedo conectarme a la VPN desde casa. "
        "Tengo una reunión urgente con cliente en 30 minutos y estoy bloqueado."
    )

    print("===== Workflow con custom logic =====")

    final_output: list[Message] | None = None

    async for event in workflow.run(user_input, stream=True):
        if event.type == "progress":
            print(f"[PROGRESS] {event.data}")

        elif event.type == "risk_decision":
            print(f"[RISK DECISION] {event.data}")

        elif event.type == "output":
            final_output = event.data

    if not final_output:
        raise RuntimeError("El workflow no produjo salida final.")

    print("\n===== Conversación final =====")

    for i, message in enumerate(final_output, start=1):
        author = (
            getattr(message, "author_name", None)
            or getattr(message, "role", None)
            or "unknown"
        )

        text = (
            getattr(message, "text", None)
            or getattr(message, "content", None)
            or ""
        )

        print(f"\n--- Mensaje {i} | {author} ---")
        print(text)


if __name__ == "__main__":
    asyncio.run(main())