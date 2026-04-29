import asyncio
import os
import re
import time
from collections.abc import Awaitable, Callable
from typing import Annotated

from dotenv import load_dotenv
from pydantic import Field

from agent_framework import (
    Agent,
    AgentContext,
    AgentMiddleware,
    AgentResponse,
    FunctionInvocationContext,
    FunctionMiddleware,
    Message,
    tool,
)

from build_client import build_maf_responses_client


load_dotenv("../.env")


# ------------------------------------------------------------
# Tool dummy de negocio
# ------------------------------------------------------------

@tool(approval_mode="never_require")
def create_support_ticket(
    category: Annotated[
        str,
        Field(description="Categoría de la incidencia: vpn, correo, permisos, facturacion u otra."),
    ],
    priority: Annotated[
        str,
        Field(description="Prioridad de la incidencia: baja, media o alta."),
    ],
    summary: Annotated[
        str,
        Field(description="Resumen breve de la incidencia."),
    ],
) -> str:
    """
    Simula la creación de un ticket.
    En un caso real llamaría a JIRA, ServiceNow, Dynamics, Zendesk, etc.
    """
    return (
        "Ticket creado correctamente.\n"
        "ID: INC-DUMMY-10042\n"
        f"Categoría: {category}\n"
        f"Prioridad: {priority}\n"
        f"Resumen: {summary}"
    )


# ------------------------------------------------------------
# Agent middleware: seguridad antes de invocar el modelo
# ------------------------------------------------------------

class SecurityAgentMiddleware(AgentMiddleware):
    """
    Middleware de seguridad a nivel de agente.

    Se ejecuta antes de que el agente llame al modelo.
    Si detecta secretos o credenciales en el mensaje del usuario,
    corta la ejecución y devuelve una respuesta segura.
    """

    SECRET_PATTERNS = [
        r"(?i)\bpassword\b\s*[:=]\s*\S+",
        r"(?i)\bcontraseña\b\s*[:=]\s*\S+",
        r"(?i)\bapi[_-]?key\b\s*[:=]\s*\S+",
        r"(?i)\bclient[_-]?secret\b\s*[:=]\s*\S+",
        r"(?i)\bsecret\b\s*[:=]\s*\S+",
        r"(?i)\btoken\b\s*[:=]\s*\S+",
    ]

    async def process(
        self,
        context: AgentContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        last_message = context.messages[-1] if context.messages else None
        text = getattr(last_message, "text", "") if last_message else ""

        if any(re.search(pattern, text or "") for pattern in self.SECRET_PATTERNS):
            print("[SecurityAgentMiddleware] Bloqueado: posible secreto detectado.")

            context.result = AgentResponse(
                messages=[
                    Message(
                        "assistant",
                        [
                            "He detectado posible información sensible en tu mensaje. "
                            "Por seguridad, no voy a procesarla. Elimina contraseñas, tokens "
                            "o secretos y vuelve a intentarlo."
                        ],
                    )
                ]
            )
            return

        print("[SecurityAgentMiddleware] Validación de seguridad superada.")
        await call_next()


# ------------------------------------------------------------
# Agent middleware: auditoría de ejecución
# ------------------------------------------------------------

class AuditAgentMiddleware(AgentMiddleware):
    """
    Middleware de auditoría a nivel de agente.

    Mide duración total, registra inicio/fin y permite centralizar
    logging sin contaminar las instrucciones del agente.
    """

    async def process(
        self,
        context: AgentContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        start = time.perf_counter()

        agent_name = getattr(context.agent, "name", "unknown-agent")
        message_count = len(context.messages or [])

        print(f"[AuditAgentMiddleware] Inicio agent={agent_name}, mensajes={message_count}")

        try:
            await call_next()
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            print(
                f"[AuditAgentMiddleware] ERROR agent={agent_name}, "
                f"duration_ms={duration_ms:.2f}, error={exc}"
            )
            raise

        duration_ms = (time.perf_counter() - start) * 1000
        print(f"[AuditAgentMiddleware] Fin agent={agent_name}, duration_ms={duration_ms:.2f}")


# ------------------------------------------------------------
# Function middleware: auditoría y control de tools
# ------------------------------------------------------------

class ToolAuditMiddleware(FunctionMiddleware):
    """
    Middleware a nivel de tool/function.

    Se ejecuta cada vez que el agente invoca una tool.
    Permite validar, auditar, medir latencia o incluso bloquear tools.
    """

    ALLOWED_TOOLS = {
        "create_support_ticket",
    }

    async def process(
        self,
        context: FunctionInvocationContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        function_name = context.function.name

        if function_name not in self.ALLOWED_TOOLS:
            print(f"[ToolAuditMiddleware] Tool bloqueada: {function_name}")
            context.result = f"Tool no permitida: {function_name}"
            context.terminate = True
            return

        print(f"[ToolAuditMiddleware] Ejecutando tool={function_name}")
        print(f"[ToolAuditMiddleware] Argumentos={context.kwargs}")

        start = time.perf_counter()

        await call_next()

        duration_ms = (time.perf_counter() - start) * 1000

        print(
            f"[ToolAuditMiddleware] Tool finalizada={function_name}, "
            f"duration_ms={duration_ms:.2f}"
        )
        print(f"[ToolAuditMiddleware] Resultado={context.result}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

async def main() -> None:
    client = build_maf_responses_client(
        endpoint=os.getenv("AOAI_ENDPOINT_SECONDARY"),
        model=os.getenv("AOAI_DEPLOYMENT_CHAT_SECONDARY", "gpt-5.1-chat"),
        api_key=os.getenv("AOAI_API_KEY_SECONDARY"),
        api_version=os.getenv("AZURE_OPENAI_RESPONSES_API_VERSION", "preview"),
    )

    agent = Agent(
        client=client,
        name="SupportAgent",
        instructions=(
            "Eres un agente de soporte técnico de nivel 1. "
            "Clasifica la incidencia, estima prioridad y, si procede, crea un ticket "
            "usando la tool create_support_ticket. "
            "No inventes IDs de ticket: el ID solo puede venir de la tool."
        ),
        tools=[
            create_support_ticket,
        ],
        middleware=[
            SecurityAgentMiddleware(),
            AuditAgentMiddleware(),
            ToolAuditMiddleware(),
        ],
    )

    print("\n===== Caso normal =====")
    result = await agent.run(
        "No puedo conectarme a la VPN desde casa y tengo una reunión con cliente en 30 minutos."
    )
    print("\nRespuesta agente:")
    print(result.text)

    print("\n===== Caso bloqueado por middleware =====")
    result = await agent.run(
        "No puedo entrar. Mi password=SuperSecreta123, ¿puedes revisar qué pasa?"
    )
    print("\nRespuesta agente:")
    print(result.text)


if __name__ == "__main__":
    asyncio.run(main())