import os
import json
from typing import List
from pydantic import BaseModel, Field, ValidationError

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI


class Source(BaseModel):
    source_id: str
    title: str
    excerpt: str


class SupportedClaim(BaseModel):
    claim: str = Field(description="Afirmación realizada en la respuesta")
    source_ids: List[str] = Field(description="Fuentes que soportan la afirmación")


class RagAnswer(BaseModel):
    answer: str
    supported_claims: List[SupportedClaim]
    insufficient_evidence: bool
    sources_used: List[str]


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Falta la variable de entorno: {name}")
    return value


credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)

token_provider = get_bearer_token_provider(
    credential,
    "https://cognitiveservices.azure.com/.default",
)

client = AzureOpenAI(
    azure_endpoint=require_env("AOAI_ENDPOINT_PRIMARY"),
    api_version="2024-10-21",
    azure_ad_token_provider=token_provider,
)


def build_context(sources: List[Source]) -> str:
    blocks = []

    for source in sources:
        blocks.append(
            f"""
[SOURCE_ID: {source.source_id}]
[TITLE: {source.title}]
[EXCERPT]
{source.excerpt}
""".strip()
        )

    return "\n\n---\n\n".join(blocks)


def generate_grounded_answer(question: str, sources: List[Source]) -> RagAnswer:
    context = build_context(sources)

    system_prompt = """
Eres un asistente RAG corporativo.

Debes responder SOLO usando el CONTEXTO.
No añadas información que no esté soportada por las fuentes.
Si falta evidencia, marca insufficient_evidence=true.

Devuelve exclusivamente JSON válido con esta forma:
{
  "answer": "...",
  "supported_claims": [
    {
      "claim": "...",
      "source_ids": ["..."]
    }
  ],
  "insufficient_evidence": false,
  "sources_used": ["..."]
}
"""

    user_prompt = f"""
PREGUNTA:
{question}

CONTEXTO:
{context}
"""

    response = client.chat.completions.create(
        model=require_env("AOAI_DEPLOYMENT_CHAT_PRIMARY"),
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.0,
        max_tokens=700,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content

    try:
        parsed = json.loads(raw)
        return RagAnswer(**parsed)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise RuntimeError(f"La respuesta no cumple el contrato esperado: {exc}") from exc


def validate_sources(answer: RagAnswer, available_sources: List[Source]) -> list[str]:
    """
    Validador simple:
    - Comprueba que todas las fuentes citadas existen.
    - Comprueba que cada claim tiene al menos una fuente.
    - No prueba equivalencia semántica profunda, pero detecta errores estructurales.
    """

    errors = []
    valid_source_ids = {s.source_id for s in available_sources}

    for source_id in answer.sources_used:
        if source_id not in valid_source_ids:
            errors.append(f"Fuente inexistente citada: {source_id}")

    for claim in answer.supported_claims:
        if not claim.source_ids:
            errors.append(f"Afirmación sin fuente: {claim.claim}")

        for source_id in claim.source_ids:
            if source_id not in valid_source_ids:
                errors.append(f"Afirmación cita fuente inexistente: {source_id}")

    if answer.insufficient_evidence and answer.supported_claims:
        errors.append("La respuesta marca evidencia insuficiente pero incluye claims soportados.")

    return errors


if __name__ == "__main__":
    question = "¿Puedo reclamar un taxi si vuelvo tarde de una visita a cliente?"

    sources = [
        Source(
            source_id="POL-GASTOS-2026-SEC-4",
            title="Política de gastos 2026 - Transporte",
            excerpt=(
                "Los gastos de transporte derivados de visitas a cliente podrán "
                "ser reembolsados cuando estén justificados y aprobados por el responsable directo. "
                "El uso de taxi requiere justificación del motivo y recibo válido."
            ),
        ),
        Source(
            source_id="POL-GASTOS-2026-SEC-7",
            title="Política de gastos 2026 - Documentación requerida",
            excerpt=(
                "Toda solicitud de reembolso debe incluir recibo, fecha, motivo del desplazamiento "
                "y centro de coste asociado."
            ),
        ),
    ]

    answer = generate_grounded_answer(question, sources)
    errors = validate_sources(answer, sources)

    print("=== RESPUESTA ===")
    print(answer.answer)
    print()

    print("=== CLAIMS SOPORTADOS ===")
    for claim in answer.supported_claims:
        print(f"- {claim.claim} -> {claim.source_ids}")

    print()
    print("=== VALIDACIÓN ===")
    if errors:
        print("Errores encontrados:")
        for error in errors:
            print(f"- {error}")
    else:
        print("Validación estructural correcta.")