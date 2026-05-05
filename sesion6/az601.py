from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(
    title="Copilot Studio Advanced Backend",
    version="1.0.0",
    description=(
        "Backend pro-code para exponer capacidades avanzadas a Copilot Studio: "
        "triage de incidencias y knowledge custom controlado."
    ),
)


# ---------------------------------------------------------------------
# Seguridad simple para laboratorio
# En producción: Entra ID, OAuth, API Management, mTLS o validación JWT.
# ---------------------------------------------------------------------

BACKEND_SHARED_SECRET = os.getenv("BACKEND_SHARED_SECRET", "dev-secret")


def require_backend_secret(x_backend_secret: str | None) -> None:
    if x_backend_secret != BACKEND_SHARED_SECRET:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized backend call",
        )


def safe_log(event: str, payload: dict) -> None:
    """
    Logging seguro para laboratorio.
    No imprimir secretos ni payloads completos con PII.
    """
    redacted = {}

    for key, value in payload.items():
        lowered = key.lower()

        if lowered in {"authorization", "token", "secret", "password", "api_key"}:
            redacted[key] = "[OCULTO]"
        else:
            redacted[key] = value

    print(
        json.dumps(
            {
                "event": event,
                "payload": redacted,
                "ts": time.time(),
            },
            ensure_ascii=False,
        )
    )


# ---------------------------------------------------------------------
# Endpoint 1: action de triage para Copilot Studio
# ---------------------------------------------------------------------

Priority = Literal["baja", "media", "alta"]
Category = Literal["vpn", "identidad", "correo", "aplicacion", "desconocido"]


class TriageRequest(BaseModel):
    user_id: str = Field(..., min_length=3)
    project_key: str | None = Field(
        default=None,
        description="Clave del proyecto si Copilot Studio ya la conoce.",
    )
    message: str = Field(..., min_length=5, max_length=2000)


class TriageResponse(BaseModel):
    category: Category
    priority: Priority
    project_key: str | None
    recommended_action: str
    missing_fields: list[str]
    safe_to_create_ticket: bool


def classify_message(message: str) -> tuple[Category, Priority]:
    lower = message.lower()

    if "vpn" in lower or "809" in lower:
        return "vpn", "alta"

    if "contraseña" in lower or "password" in lower or "mfa" in lower:
        return "identidad", "media"

    if "correo" in lower or "outlook" in lower:
        return "correo", "media"

    if "aplicación" in lower or "aplicacion" in lower or "error 500" in lower:
        return "aplicacion", "alta"

    return "desconocido", "baja"


@app.post("/triage", response_model=TriageResponse)
async def triage_incident(
    body: TriageRequest,
    x_backend_secret: str | None = Header(default=None),
) -> TriageResponse:
    require_backend_secret(x_backend_secret)

    category, priority = classify_message(body.message)

    missing_fields = []

    if not body.project_key:
        missing_fields.append("project_key")

    if category == "desconocido":
        missing_fields.append("categoria_clara")

    safe_to_create_ticket = len(missing_fields) == 0

    if safe_to_create_ticket:
        recommended_action = (
            f"Crear ticket de prioridad {priority} para el proyecto {body.project_key}."
        )
    else:
        recommended_action = (
            "Pedir los datos faltantes antes de crear el ticket: "
            + ", ".join(missing_fields)
        )

    response = TriageResponse(
        category=category,
        priority=priority,
        project_key=body.project_key,
        recommended_action=recommended_action,
        missing_fields=missing_fields,
        safe_to_create_ticket=safe_to_create_ticket,
    )

    safe_log(
        "triage_incident",
        {
            "user_id": body.user_id,
            "project_key": body.project_key,
            "category": category,
            "priority": priority,
            "safe_to_create_ticket": safe_to_create_ticket,
        },
    )

    return response


# ---------------------------------------------------------------------
# Endpoint 2: custom knowledge controlado
# ---------------------------------------------------------------------

class KnowledgeRequest(BaseModel):
    search_query: str = Field(..., min_length=3, max_length=1000)
    user_id: str = Field(..., min_length=3)
    project_key: str | None = None
    max_results: int = Field(default=5, ge=1, le=15)


class KnowledgeSnippet(BaseModel):
    title: str
    content: str
    url: str | None = None
    source: str
    score: float


class KnowledgeResponse(BaseModel):
    snippets: list[KnowledgeSnippet]
    applied_filters: dict[str, str | None]


@dataclass
class FakeDocument:
    title: str
    project_key: str
    content: str
    url: str
    source: str


FAKE_CORPUS = [
    FakeDocument(
        title="Guía VPN ALFA",
        project_key="ALFA",
        content=(
            "Ante error VPN-809, revisar certificado del dispositivo, "
            "firewall local y conectividad hacia la red interna."
        ),
        url="https://contoso.example/docs/alfa/vpn",
        source="sharepoint-alfa",
    ),
    FakeDocument(
        title="Guía Identidad ALFA",
        project_key="ALFA",
        content=(
            "Para problemas de MFA, verificar registro del método, hora del dispositivo "
            "y estado de la cuenta. Soporte no debe pedir contraseñas."
        ),
        url="https://contoso.example/docs/alfa/identidad",
        source="sharepoint-alfa",
    ),
    FakeDocument(
        title="Guía VPN BETA",
        project_key="BETA",
        content=(
            "En BETA, los errores VPN se escalan directamente al equipo de redes "
            "si afectan a más de cinco usuarios."
        ),
        url="https://contoso.example/docs/beta/vpn",
        source="sharepoint-beta",
    ),
]


def score_document(query: str, doc: FakeDocument) -> float:
    query_terms = set(query.lower().split())
    text_terms = set((doc.title + " " + doc.content).lower().split())

    if not query_terms:
        return 0.0

    overlap = query_terms.intersection(text_terms)
    return round(len(overlap) / len(query_terms), 3)


@app.post("/knowledge", response_model=KnowledgeResponse)
async def custom_knowledge(
    body: KnowledgeRequest,
    x_backend_secret: str | None = Header(default=None),
) -> KnowledgeResponse:
    require_backend_secret(x_backend_secret)

    # Control de alcance: si hay project_key, no devolvemos documentos de otros proyectos.
    candidates = [
        doc for doc in FAKE_CORPUS
        if body.project_key is None or doc.project_key == body.project_key
    ]

    scored = [
        (doc, score_document(body.search_query, doc))
        for doc in candidates
    ]

    scored = [
        item for item in scored
        if item[1] > 0
    ]

    scored.sort(key=lambda item: item[1], reverse=True)

    snippets = [
        KnowledgeSnippet(
            title=doc.title,
            content=doc.content,
            url=doc.url,
            source=doc.source,
            score=score,
        )
        for doc, score in scored[: body.max_results]
    ]

    safe_log(
        "custom_knowledge",
        {
            "user_id": body.user_id,
            "project_key": body.project_key,
            "query_len": len(body.search_query),
            "results": len(snippets),
        },
    )

    return KnowledgeResponse(
        snippets=snippets,
        applied_filters={
            "project_key": body.project_key,
        },
    )


# ---------------------------------------------------------------------
# Healthcheck
# ---------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "service": "copilot-studio-advanced-backend",
        "version": "1.0.0",
    }