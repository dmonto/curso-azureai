import os
import re
import json
import time
import uuid
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import AzureOpenAI
from pydantic import BaseModel, Field

load_dotenv("../.env")


CACHE_FILE = Path("agent_cache.json")
EVENTS_FILE = Path("agent_cache_events.jsonl")


# ---------------------------------------------------------------------
# 1. Configuración de versiones
# ---------------------------------------------------------------------

CORPUS_VERSION = "corpus-2026-05-02"
PROMPT_VERSION = "prompt-2026-05-02"
POLICY_VERSION = "policy-2.1.0"


# ---------------------------------------------------------------------
# 2. Modelos
# ---------------------------------------------------------------------

class CacheEntry(BaseModel):
    key: str
    value: Any
    created_at: float
    ttl_seconds: int
    tenant_id: str
    domain: str
    cache_type: str
    corpus_version: str
    prompt_version: str
    policy_version: str


class CacheEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tenant_id: str
    domain: str
    cache_type: str
    cache_hit: bool
    latency_ms: int
    estimated_saved_cost: float = 0.0
    reason: Optional[str] = None


# ---------------------------------------------------------------------
# 3. Cliente Azure OpenAI
# ---------------------------------------------------------------------

def build_client() -> AzureOpenAI:
    endpoint = os.getenv("AOAI_ENDPOINT_SECONDARY")
    api_key = os.getenv("AOAI_API_KEY_SECONDARY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

    if not endpoint:
        raise RuntimeError("Falta AZURE_OPENAI_ENDPOINT en .env")

    if not api_key:
        raise RuntimeError("Falta AZURE_OPENAI_API_KEY en .env")

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )


# ---------------------------------------------------------------------
# 4. Persistencia simple de cache
# ---------------------------------------------------------------------

def load_cache() -> Dict[str, CacheEntry]:
    if not CACHE_FILE.exists():
        return {}

    raw = json.loads(CACHE_FILE.read_text(encoding="utf-8"))

    return {
        key: CacheEntry(**value)
        for key, value in raw.items()
    }


def save_cache(cache: Dict[str, CacheEntry]) -> None:
    serializable = {
        key: entry.model_dump()
        for key, entry in cache.items()
    }

    CACHE_FILE.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_event(event: CacheEvent) -> None:
    with EVENTS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event.model_dump(), ensure_ascii=False) + "\n")


def is_entry_valid(entry: CacheEntry) -> bool:
    now = time.time()

    if now - entry.created_at > entry.ttl_seconds:
        return False

    if entry.corpus_version != CORPUS_VERSION:
        return False

    if entry.prompt_version != PROMPT_VERSION:
        return False

    if entry.policy_version != POLICY_VERSION:
        return False

    return True


# ---------------------------------------------------------------------
# 5. Cache key segura
# ---------------------------------------------------------------------

def normalize_question(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[¿?¡!.,;:]", "", text)
    return text


def sha256_short(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_cache_key(
    cache_type: str,
    tenant_id: str,
    domain: str,
    user_role: str,
    normalized_question: str,
    extra: Optional[str] = None,
) -> str:
    raw = "|".join(
        [
            cache_type,
            tenant_id,
            domain,
            user_role,
            CORPUS_VERSION,
            PROMPT_VERSION,
            POLICY_VERSION,
            normalized_question,
            extra or "",
        ]
    )

    return sha256_short(raw)


# ---------------------------------------------------------------------
# 6. Coste estimado
# ---------------------------------------------------------------------

def env_float(name: str, default: str) -> float:
    return float(os.getenv(name, default))


PRICE_INPUT = env_float("PRICE_INPUT_PER_1M_TOKENS", "2.50")
PRICE_OUTPUT = env_float("PRICE_OUTPUT_PER_1M_TOKENS", "10.00")


def rough_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    input_cost = (input_tokens / 1_000_000) * PRICE_INPUT
    output_cost = (output_tokens / 1_000_000) * PRICE_OUTPUT
    return round(input_cost + output_cost, 6)


# ---------------------------------------------------------------------
# 7. Corpus local de laboratorio
# ---------------------------------------------------------------------

LOCAL_DOCS = [
    {
        "tenant_id": "civica",
        "domain": "soporte",
        "title": "Runbook VPN Cívica",
        "content": "Para problemas VPN, revisar MFA, túnel SSL, grupo de acceso y conectividad interna.",
        "source": "sharepoint/civica/soporte/vpn.md",
    },
    {
        "tenant_id": "globex",
        "domain": "soporte",
        "title": "Runbook VPN Globex",
        "content": "Para Globex, la revisión inicial de VPN se realiza en el portal Zero Trust.",
        "source": "sharepoint/globex/soporte/vpn.md",
    },
]


# ---------------------------------------------------------------------
# 8. Embedding cache simulado
# ---------------------------------------------------------------------

def get_or_create_embedding(
    cache: Dict[str, CacheEntry],
    tenant_id: str,
    domain: str,
    user_role: str,
    question: str,
) -> List[float]:
    normalized = normalize_question(question)

    key = build_cache_key(
        cache_type="embedding",
        tenant_id=tenant_id,
        domain=domain,
        user_role=user_role,
        normalized_question=normalized,
    )

    start = time.perf_counter()

    entry = cache.get(key)
    if entry and is_entry_valid(entry):
        latency_ms = int((time.perf_counter() - start) * 1000)
        write_event(
            CacheEvent(
                tenant_id=tenant_id,
                domain=domain,
                cache_type="embedding",
                cache_hit=True,
                latency_ms=latency_ms,
                estimated_saved_cost=0.0001,
            )
        )
        return entry.value

    # Simulación determinista de embedding
    digest = hashlib.sha256(normalized.encode("utf-8")).digest()
    embedding = [round(byte / 255, 4) for byte in digest[:8]]

    cache[key] = CacheEntry(
        key=key,
        value=embedding,
        created_at=time.time(),
        ttl_seconds=24 * 3600,
        tenant_id=tenant_id,
        domain=domain,
        cache_type="embedding",
        corpus_version=CORPUS_VERSION,
        prompt_version=PROMPT_VERSION,
        policy_version=POLICY_VERSION,
    )

    latency_ms = int((time.perf_counter() - start) * 1000)
    write_event(
        CacheEvent(
            tenant_id=tenant_id,
            domain=domain,
            cache_type="embedding",
            cache_hit=False,
            latency_ms=latency_ms,
            reason="miss_or_invalid",
        )
    )

    return embedding


# ---------------------------------------------------------------------
# 9. Retrieval cache
# ---------------------------------------------------------------------

def search_docs_local(
    tenant_id: str,
    domain: str,
    question: str,
    top_k: int = 2,
) -> List[Dict[str, str]]:
    normalized = normalize_question(question)
    terms = set(normalized.split())

    docs = []

    for doc in LOCAL_DOCS:
        if doc["tenant_id"] != tenant_id:
            continue

        if doc["domain"] != domain:
            continue

        searchable = normalize_question(doc["title"] + " " + doc["content"])

        if any(term in searchable for term in terms):
            docs.append(doc)

    return docs[:top_k]


def get_or_create_retrieval(
    cache: Dict[str, CacheEntry],
    tenant_id: str,
    domain: str,
    user_role: str,
    question: str,
) -> List[Dict[str, str]]:
    normalized = normalize_question(question)

    key = build_cache_key(
        cache_type="retrieval",
        tenant_id=tenant_id,
        domain=domain,
        user_role=user_role,
        normalized_question=normalized,
        extra="top_k=2",
    )

    start = time.perf_counter()

    entry = cache.get(key)
    if entry and is_entry_valid(entry):
        latency_ms = int((time.perf_counter() - start) * 1000)
        write_event(
            CacheEvent(
                tenant_id=tenant_id,
                domain=domain,
                cache_type="retrieval",
                cache_hit=True,
                latency_ms=latency_ms,
                estimated_saved_cost=0.0005,
            )
        )
        return entry.value

    docs = search_docs_local(
        tenant_id=tenant_id,
        domain=domain,
        question=question,
    )

    cache[key] = CacheEntry(
        key=key,
        value=docs,
        created_at=time.time(),
        ttl_seconds=10 * 60,
        tenant_id=tenant_id,
        domain=domain,
        cache_type="retrieval",
        corpus_version=CORPUS_VERSION,
        prompt_version=PROMPT_VERSION,
        policy_version=POLICY_VERSION,
    )

    latency_ms = int((time.perf_counter() - start) * 1000)
    write_event(
        CacheEvent(
            tenant_id=tenant_id,
            domain=domain,
            cache_type="retrieval",
            cache_hit=False,
            latency_ms=latency_ms,
            reason="miss_or_invalid",
        )
    )

    return docs


# ---------------------------------------------------------------------
# 10. Respuesta final cacheada con cautela
# ---------------------------------------------------------------------

def is_safe_to_cache_final_answer(
    tenant_id: str,
    domain: str,
    user_role: str,
    docs: List[Dict[str, str]],
) -> bool:
    """
    Regla simplificada:
    - Solo cacheamos respuestas de soporte.
    - No cacheamos si el rol no es estándar.
    - No cacheamos si no hay documentos.
    En producción se añadiría sensibilidad, PII, permisos y tipo de respuesta.
    """
    if domain != "soporte":
        return False

    if user_role not in {"support_l1", "employee"}:
        return False

    if not docs:
        return False

    return True


def build_context(docs: List[Dict[str, str]]) -> str:
    if not docs:
        return "SIN_CONTEXTO"

    blocks = []

    for i, doc in enumerate(docs, start=1):
        blocks.append(
            f"[DOC {i}]\n"
            f"Título: {doc['title']}\n"
            f"Fuente: {doc['source']}\n"
            f"Contenido: {doc['content']}"
        )

    return "\n\n".join(blocks)


def get_cached_final_answer(
    cache: Dict[str, CacheEntry],
    tenant_id: str,
    domain: str,
    user_role: str,
    question: str,
) -> Optional[str]:
    normalized = normalize_question(question)

    key = build_cache_key(
        cache_type="final_answer",
        tenant_id=tenant_id,
        domain=domain,
        user_role=user_role,
        normalized_question=normalized,
    )

    start = time.perf_counter()
    entry = cache.get(key)

    if entry and is_entry_valid(entry):
        latency_ms = int((time.perf_counter() - start) * 1000)
        write_event(
            CacheEvent(
                tenant_id=tenant_id,
                domain=domain,
                cache_type="final_answer",
                cache_hit=True,
                latency_ms=latency_ms,
                estimated_saved_cost=0.01,
            )
        )
        return entry.value

    return None


def store_final_answer(
    cache: Dict[str, CacheEntry],
    tenant_id: str,
    domain: str,
    user_role: str,
    question: str,
    answer: str,
) -> None:
    normalized = normalize_question(question)

    key = build_cache_key(
        cache_type="final_answer",
        tenant_id=tenant_id,
        domain=domain,
        user_role=user_role,
        normalized_question=normalized,
    )

    cache[key] = CacheEntry(
        key=key,
        value=answer,
        created_at=time.time(),
        ttl_seconds=5 * 60,
        tenant_id=tenant_id,
        domain=domain,
        cache_type="final_answer",
        corpus_version=CORPUS_VERSION,
        prompt_version=PROMPT_VERSION,
        policy_version=POLICY_VERSION,
    )


# ---------------------------------------------------------------------
# 11. Ejecución del agente con cache
# ---------------------------------------------------------------------

def answer_with_cache(
    tenant_id: str,
    domain: str,
    user_role: str,
    question: str,
) -> str:
    cache = load_cache()

    cached_answer = get_cached_final_answer(
        cache=cache,
        tenant_id=tenant_id,
        domain=domain,
        user_role=user_role,
        question=question,
    )

    if cached_answer:
        save_cache(cache)
        return f"[CACHE HIT - FINAL]\n{cached_answer}"

    # Cache de embedding
    _embedding = get_or_create_embedding(
        cache=cache,
        tenant_id=tenant_id,
        domain=domain,
        user_role=user_role,
        question=question,
    )

    # Cache de retrieval
    docs = get_or_create_retrieval(
        cache=cache,
        tenant_id=tenant_id,
        domain=domain,
        user_role=user_role,
        question=question,
    )

    context = build_context(docs)

    system_prompt = """
Eres un agente de soporte L1.
Usa solo el contexto autorizado.
Si no hay contexto suficiente, responde: "No tengo contexto suficiente".
Da máximo 4 pasos.
Incluye fuente si existe.
""".strip()

    user_prompt = f"""
Pregunta:
{question}

Contexto autorizado:
{context}
""".strip()

    client = build_client()
    deployment = os.getenv("AOAI_DEPLOYMENT_CHAT_SECONDARY", "gpt-4o")

    start = time.perf_counter()

    response = client.chat.completions.create(
        model=deployment,
        temperature=0.2,
        max_tokens=350,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    latency_ms = int((time.perf_counter() - start) * 1000)
    answer = response.choices[0].message.content or ""

    usage = response.usage
    input_tokens = usage.prompt_tokens if usage else rough_tokens(system_prompt + user_prompt)
    output_tokens = usage.completion_tokens if usage else rough_tokens(answer)
    cost = estimate_cost(input_tokens, output_tokens)

    write_event(
        CacheEvent(
            tenant_id=tenant_id,
            domain=domain,
            cache_type="model_call",
            cache_hit=False,
            latency_ms=latency_ms,
            estimated_saved_cost=0.0,
            reason=f"model_cost={cost}",
        )
    )

    if is_safe_to_cache_final_answer(tenant_id, domain, user_role, docs):
        store_final_answer(
            cache=cache,
            tenant_id=tenant_id,
            domain=domain,
            user_role=user_role,
            question=question,
            answer=answer,
        )

    save_cache(cache)

    return answer


# ---------------------------------------------------------------------
# 12. Métricas de cache
# ---------------------------------------------------------------------

def build_cache_report() -> Dict[str, Any]:
    if not EVENTS_FILE.exists():
        return {"message": "No hay eventos de cache."}

    rows = []

    with EVENTS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    if not rows:
        return {"message": "No hay eventos."}

    total = len(rows)
    hits = sum(1 for r in rows if r["cache_hit"])
    misses = total - hits
    saved = sum(float(r.get("estimated_saved_cost", 0)) for r in rows)

    by_type: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        cache_type = row["cache_type"]

        if cache_type not in by_type:
            by_type[cache_type] = {
                "events": 0,
                "hits": 0,
                "misses": 0,
                "estimated_saved_cost": 0.0,
            }

        by_type[cache_type]["events"] += 1
        by_type[cache_type]["hits"] += 1 if row["cache_hit"] else 0
        by_type[cache_type]["misses"] += 0 if row["cache_hit"] else 1
        by_type[cache_type]["estimated_saved_cost"] += float(row.get("estimated_saved_cost", 0))

    return {
        "total_events": total,
        "hits": hits,
        "misses": misses,
        "hit_rate_pct": round((hits / total) * 100, 2),
        "estimated_saved_cost": round(saved, 6),
        "by_type": by_type,
    }


# ---------------------------------------------------------------------
# 13. Prueba
# ---------------------------------------------------------------------

if __name__ == "__main__":
    cases = [
        {
            "tenant_id": "civica",
            "domain": "soporte",
            "user_role": "support_l1",
            "question": "Tengo problemas con la VPN. ¿Qué debo revisar?",
        },
        {
            "tenant_id": "civica",
            "domain": "soporte",
            "user_role": "support_l1",
            "question": "Tengo problemas con la VPN. ¿Qué debo revisar?",
        },
        {
            "tenant_id": "globex",
            "domain": "soporte",
            "user_role": "support_l1",
            "question": "Tengo problemas con la VPN. ¿Qué debo revisar?",
        },
    ]

    for case in cases:
        print("\n" + "=" * 90)
        print(json.dumps(case, ensure_ascii=False, indent=2))
        print("=" * 90)
        print(answer_with_cache(**case))

    print("\n" + "=" * 90)
    print("CACHE REPORT")
    print("=" * 90)
    print(json.dumps(build_cache_report(), ensure_ascii=False, indent=2))