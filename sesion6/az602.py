from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph


DocumentRoute = Literal[
    "ready_to_index",
    "needs_metadata",
    "compliance_review",
    "discard",
]


class DocumentState(TypedDict):
    document_title: str
    document_text: str
    route: DocumentRoute | None
    reason: str | None
    output: str | None


def analyze_document(state: DocumentState) -> dict:
    """
    Nodo analizador.

    Lee el título y el contenido del documento y decide qué ruta seguir.

    En una versión posterior, este nodo podría llamar a Azure OpenAI
    para clasificar el documento con más precisión.
    Aquí usamos reglas deterministas para que el patrón sea fácil
    de migrar después a MAF.
    """
    title = state["document_title"].lower()
    text = state["document_text"].lower()
    combined = f"{title} {text}"

    if any(term in combined for term in ["contraseña", "password", "secreto", "token", "api key"]):
        return {
            "route": "compliance_review",
            "reason": (
                "El documento parece contener credenciales, secretos o información "
                "sensible que requiere revisión antes de indexarse."
            ),
        }

    if any(term in combined for term in ["borrador", "draft", "pendiente de validar", "no aprobado"]):
        return {
            "route": "discard",
            "reason": (
                "El documento parece ser un borrador o contenido no aprobado. "
                "No debe entrar todavía en la base de conocimiento."
            ),
        }

    if any(term in combined for term in ["sin owner", "sin responsable", "sin proyecto", "sin fecha"]):
        return {
            "route": "needs_metadata",
            "reason": (
                "El documento parece útil, pero faltan metadatos mínimos "
                "como owner, proyecto, fecha o dominio."
            ),
        }

    if any(term in combined for term in ["procedimiento", "guía", "manual", "runbook", "política"]):
        return {
            "route": "ready_to_index",
            "reason": (
                "El documento parece ser conocimiento operativo válido "
                "y puede prepararse para indexación."
            ),
        }

    return {
        "route": "needs_metadata",
        "reason": (
            "No se detecta suficiente contexto para decidir si el documento "
            "debe indexarse."
        ),
    }


def ready_to_index(state: DocumentState) -> dict:
    return {
        "output": (
            "Ruta READY_TO_INDEX.\n"
            "Acción recomendada: preparar el documento para indexación en el sistema RAG.\n"
            "Pasos sugeridos:\n"
            "- extraer título, resumen y dominio;\n"
            "- generar metadatos de proyecto;\n"
            "- dividir en chunks;\n"
            "- enviar a Azure AI Search o índice equivalente;\n"
            "- conservar referencia a la fuente original.\n"
            f"Motivo: {state['reason']}"
        )
    }


def needs_metadata(state: DocumentState) -> dict:
    return {
        "output": (
            "Ruta NEEDS_METADATA.\n"
            "Acción recomendada: pedir información adicional antes de indexar.\n"
            "Metadatos mínimos sugeridos:\n"
            "- owner del documento;\n"
            "- proyecto o dominio;\n"
            "- fecha de vigencia;\n"
            "- nivel de sensibilidad;\n"
            "- fuente original.\n"
            f"Motivo: {state['reason']}"
        )
    }


def compliance_review(state: DocumentState) -> dict:
    return {
        "output": (
            "Ruta COMPLIANCE_REVIEW.\n"
            "Acción recomendada: bloquear la indexación automática y enviar a revisión.\n"
            "El documento puede contener secretos, credenciales, datos sensibles "
            "o información que no debe exponerse en respuestas generativas.\n"
            f"Motivo: {state['reason']}"
        )
    }


def discard(state: DocumentState) -> dict:
    return {
        "output": (
            "Ruta DISCARD.\n"
            "Acción recomendada: no indexar este documento por ahora.\n"
            "Puede ser un borrador, estar pendiente de aprobación o no ser una "
            "fuente válida de conocimiento corporativo.\n"
            f"Motivo: {state['reason']}"
        )
    }


def route_after_analysis(state: DocumentState) -> DocumentRoute:
    """
    Función de routing.

    LangGraph usará el valor devuelto para decidir el siguiente nodo.
    """
    route = state.get("route")

    if route in {
        "ready_to_index",
        "needs_metadata",
        "compliance_review",
        "discard",
    }:
        return route

    return "needs_metadata"


def build_graph():
    builder = StateGraph(DocumentState)

    # Nodes
    builder.add_node("analyze_document", analyze_document)
    builder.add_node("ready_to_index", ready_to_index)
    builder.add_node("needs_metadata", needs_metadata)
    builder.add_node("compliance_review", compliance_review)
    builder.add_node("discard", discard)

    # Entry point
    builder.add_edge(START, "analyze_document")

    # Conditional routing
    builder.add_conditional_edges(
        "analyze_document",
        route_after_analysis,
        {
            "ready_to_index": "ready_to_index",
            "needs_metadata": "needs_metadata",
            "compliance_review": "compliance_review",
            "discard": "discard",
        },
    )

    # End edges
    builder.add_edge("ready_to_index", END)
    builder.add_edge("needs_metadata", END)
    builder.add_edge("compliance_review", END)
    builder.add_edge("discard", END)

    return builder.compile()


def run_case(document_title: str, document_text: str) -> None:
    graph = build_graph()

    initial_state: DocumentState = {
        "document_title": document_title,
        "document_text": document_text,
        "route": None,
        "reason": None,
        "output": None,
    }

    final_state = graph.invoke(initial_state)

    print("=" * 80)
    print("DOCUMENTO:")
    print(final_state["document_title"])
    print()
    print("RUTA:", final_state["route"])
    print("MOTIVO:", final_state["reason"])
    print()
    print("SALIDA:")
    print(final_state["output"])


if __name__ == "__main__":
    test_cases = [
        {
            "title": "Guía de conexión VPN para empleados",
            "text": (
                "Este procedimiento explica cómo configurar la VPN corporativa "
                "y resolver errores habituales de conexión."
            ),
        },
        {
            "title": "Notas internas sin owner",
            "text": (
                "Documento útil sobre despliegues, pero sin responsable, "
                "sin proyecto y sin fecha de vigencia."
            ),
        },
        {
            "title": "Credenciales temporales para pruebas",
            "text": (
                "Este documento contiene password temporal, token de API "
                "y claves para ejecutar pruebas."
            ),
        },
        {
            "title": "Borrador de política de teletrabajo",
            "text": (
                "Draft pendiente de validar por legal. No aprobado todavía "
                "para comunicación interna."
            ),
        },
    ]

    for case in test_cases:
        run_case(
            document_title=case["title"],
            document_text=case["text"],
        )