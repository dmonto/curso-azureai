import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import AzureOpenAI

from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from azure.identity import DefaultAzureCredential, get_bearer_token_provider


load_dotenv()
credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)

token_provider = get_bearer_token_provider(
    credential,
    "https://cognitiveservices.azure.com/.default"
)

# Envía telemetría a Application Insights.
# En local, si no hay connection string, el script puede seguir funcionando sin exportar.
connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
if connection_string:
    configure_azure_monitor(connection_string=connection_string)

tracer = trace.get_tracer("curso.azureai.agents")


@dataclass
class Ticket:
    title: str
    category: str
    priority: str
    description: str


def classify_intent(user_text: str) -> dict:
    with tracer.start_as_current_span(
        "classify_intent",
        attributes={
            "agent.step": "classification",
            "app.component": "triage",
        },
    ) as span:
        text = user_text.lower()

        if "vpn" in text or "conectar" in text or "acceso" in text:
            category = "soporte_tecnico"
            confidence = 0.89
        elif "factura" in text or "pago" in text:
            category = "facturacion"
            confidence = 0.82
        else:
            category = "general"
            confidence = 0.45

        span.set_attribute("agent.intent.category", category)
        span.set_attribute("agent.intent.confidence", confidence)

        return {
            "category": category,
            "confidence": confidence,
        }


def retrieve_context(category: str) -> list[str]:
    with tracer.start_as_current_span(
        "retrieval soporte_kb",
        attributes={
            "gen_ai.operation.name": "retrieval",
            "gen_ai.data_source.id": "soporte_kb",
            "agent.category": category,
        },
    ) as span:
        # Simulación de búsqueda documental.
        time.sleep(0.2)

        docs = [
            "Guía VPN: verificar MFA, credenciales y cliente actualizado.",
            "Si el error persiste, registrar ticket con sistema operativo y hora del fallo.",
        ]

        span.set_attribute("retrieval.document_count", len(docs))
        span.set_attribute("retrieval.top_score", 0.84)

        return docs


def create_ticket(ticket: Ticket) -> str:
    with tracer.start_as_current_span(
        "execute_tool create_ticket",
        attributes={
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": "create_ticket",
            "ticket.category": ticket.category,
            "ticket.priority": ticket.priority,
        },
    ) as span:
        # Simulación de llamada a Jira / ITSM.
        time.sleep(0.3)

        ticket_id = "SUP-1024"
        span.set_attribute("ticket.id", ticket_id)
        span.set_attribute("tool.success", True)

        return ticket_id


def call_model(user_text: str, context_docs: list[str]) -> str:
    deployment = os.getenv("AOAI_DEPLOYMENT_CHAT_PRIMARY")

    client = AzureOpenAI(
        azure_endpoint=os.getenv("AOAI_ENDPOINT_PRIMARY"),
        api_version="2024-10-21",
        azure_ad_token_provider=token_provider,
    )

    with tracer.start_as_current_span(
        f"chat {deployment}",
        attributes={
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": "azure_openai",
            "gen_ai.request.model": deployment,
            "agent.step": "model_response",
        },
    ) as span:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un agente de soporte de nivel 1. "
                        "Responde de forma breve y operativa. "
                        "Usa solo el contexto proporcionado."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Consulta del usuario: {user_text}\n\n"
                        f"Contexto disponible:\n- " + "\n- ".join(context_docs)
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=250,
        )

        usage = response.usage
        if usage:
            span.set_attribute("gen_ai.usage.input_tokens", usage.prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", usage.completion_tokens)
            span.set_attribute("gen_ai.usage.total_tokens", usage.total_tokens)

        return response.choices[0].message.content


def run_support_agent(user_text: str) -> str:
    with tracer.start_as_current_span(
        "invoke_agent soporte_n1",
        attributes={
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": "soporte_n1",
            "app.scenario": "support_l1",
        },
    ) as span:
        try:
            classification = classify_intent(user_text)
            context_docs = retrieve_context(classification["category"])
            answer = call_model(user_text, context_docs)

            if classification["category"] == "soporte_tecnico":
                ticket = Ticket(
                    title="Incidencia de acceso VPN",
                    category=classification["category"],
                    priority="media",
                    description=user_text,
                )
                ticket_id = create_ticket(ticket)
                span.set_attribute("agent.ticket_created", True)
                span.set_attribute("agent.ticket_id", ticket_id)
                return f"{answer}\n\nTicket creado: {ticket_id}"

            span.set_attribute("agent.ticket_created", False)
            return answer

        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


if __name__ == "__main__":
    result = run_support_agent(
        "No puedo conectarme a la VPN desde casa desde esta mañana."
    )
    print(result)