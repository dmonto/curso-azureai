import os
from dataclasses import dataclass
from typing import List

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI
from dotenv import load_dotenv

from agent_framework import Agent
from build_client import build_maf_responses_client

@dataclass
class RetrievedChunk:
    title: str
    content: str
    source: str
    score: float
    reranker_score: float | None = None

load_dotenv("../.env")

def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Falta la variable de entorno: {name}")
    return value


credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)

token_provider = get_bearer_token_provider(
    credential,
    "https://cognitiveservices.azure.com/.default"
)

openai_client = AzureOpenAI(
    azure_endpoint=require_env("AOAI_ENDPOINT_PRIMARY"),
    api_version="2024-10-21",
    azure_ad_token_provider=token_provider,
)

search_client = SearchClient(
    endpoint=require_env("AZURE_SEARCH_ENDPOINT"),
    index_name=require_env("AZURE_SEARCH_INDEX"),
    credential=credential,
)


def embed_query(query: str) -> list[float]:
    response = openai_client.embeddings.create(
        model=require_env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        input=query,
    )
    return response.data[0].embedding


def retrieve_candidates(
    query: str,
    top: int = 12,
    vector_k: int = 20,
    domain_filter: str | None = None,
) -> List[RetrievedChunk]:
    """
    Recupera candidatos usando búsqueda híbrida:
    - search_text para keyword/BM25
    - vector_queries para similitud semántica
    - semantic ranker si el índice tiene configuración semántica
    """

    query_vector = embed_query(query)

    vector_query = VectorizedQuery(
        vector=query_vector,
        k=vector_k,
        fields="contentVector",
    )

    filter_expression = None
    if domain_filter:
        filter_expression = f"domain eq '{domain_filter}'"

    results = search_client.search(
        search_text=query,
        vector_queries=[vector_query],
        filter=filter_expression,
        query_type="semantic",
        semantic_configuration_name=os.getenv("AZURE_SEARCH_SEMANTIC_CONFIG", "default"),
        query_caption="extractive",
        query_answer="extractive",
        top=top,
        select=["title", "content", "source", "domain"],
    )

    chunks: List[RetrievedChunk] = []

    for item in results:
        score = item.get("@search.score", 0.0)
        reranker_score = (
            item.get("@search.rerankerScore")
            or item.get("@search.reranker_score")
        )

        chunks.append(
            RetrievedChunk(
                title=item.get("title", "Sin título"),
                content=item.get("content", ""),
                source=item.get("source", "desconocido"),
                score=float(score or 0.0),
                reranker_score=float(reranker_score) if reranker_score else None,
            )
        )

    return chunks


def select_context(
    chunks: List[RetrievedChunk],
    max_chunks: int = 5,
    max_chars: int = 6000,
    min_score: float = 0.01,
) -> str:
    """
    Construye un contexto final con presupuesto.
    En producción, convendría medir tokens reales con tiktoken u otra librería.
    Aquí usamos caracteres para simplificar el laboratorio.
    """

    # Priorizamos reranker_score cuando exista; si no, score normal.
    sorted_chunks = sorted(
        chunks,
        key=lambda c: c.reranker_score if c.reranker_score is not None else c.score,
        reverse=True,
    )

    selected = []
    used_chars = 0

    for chunk in sorted_chunks:
        if chunk.score < min_score:
            continue

        block = (
            f"[Fuente: {chunk.source}]\n"
            f"[Título: {chunk.title}]\n"
            f"{chunk.content.strip()}\n"
        )

        if used_chars + len(block) > max_chars:
            continue

        selected.append(block)
        used_chars += len(block)

        if len(selected) >= max_chunks:
            break

    return "\n---\n".join(selected)

def build_rag_answer_agent() -> Agent:
    client = build_maf_responses_client(
        endpoint=require_env("AOAI_ENDPOINT_PRIMARY"),
        model=require_env("AOAI_DEPLOYMENT_CHAT_PRIMARY"),
        api_key=os.getenv("AOAI_API_KEY_PRIMARY"),
        api_version=os.getenv("AZURE_OPENAI_RESPONSES_API_VERSION", "preview"),
    )

    return Agent(
        client=client,
        name="rag_answer_agent",
        description="Agente que responde preguntas usando exclusivamente contexto RAG.",
        instructions=(
            "Eres un asistente corporativo basado en RAG.\n"
            "Responde únicamente con la información del CONTEXTO.\n"
            "Si el contexto no contiene la respuesta, di exactamente: "
            "\"No encuentro información suficiente en las fuentes recuperadas\".\n"
            "Incluye al final una sección breve llamada \"Fuentes usadas\".\n"
            "No uses conocimiento general si no está soportado por el contexto."
        ),
    )

async def answer_with_context_maf(
    question: str,
    context: str,
    agent: Agent | None = None,
) -> str:
    rag_agent = agent or build_rag_answer_agent()

    user_prompt = f"""
PREGUNTA:
{question}

CONTEXTO:
{context}
""".strip()

    result = await rag_agent.run(user_prompt)

    return result.text

if __name__ == "__main__":
    load_dotenv()
    question = "¿Como compongo un hand-off?"

    candidates = retrieve_candidates(
        query=question,
        top=12,
        vector_k=20,
        domain_filter="curso",
    )

    context = select_context(
        candidates,
        max_chunks=5,
        max_chars=6000,
        min_score=0.01,
    )

    print("=== CONTEXTO FINAL ===")
    print(context)
    print()

    print("=== RESPUESTA ===")
    print(answer_with_context_maf(question, context))
