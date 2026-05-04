import json
import os
import statistics
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from typing import Any

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAIError
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from azure.ai.evaluation import (
    AzureOpenAIModelConfiguration,
    GroundednessEvaluator,
    RelevanceEvaluator,
    CoherenceEvaluator,
    FluencyEvaluator,
    SimilarityEvaluator,
)

load_dotenv()


DATASET = [
    {
        "id": "vpn_809",
        "category": "soporte",
        "query": "¿Qué hago si falla la VPN con error 809?",
        "context": (
            "Guía VPN 2.1. Ante error VPN-809, revisar primero el certificado "
            "del dispositivo y después el firewall local. Si persiste, escalar a soporte L2."
        ),
        "ground_truth": (
            "Debe revisarse primero el certificado del dispositivo y después el firewall local. "
            "Si el problema continúa, se escala a soporte L2."
        ),
        "expected_sources": ["GUIA-VPN-2.1"],
    },
    {
        "id": "gastos_taxi",
        "category": "finanzas",
        "query": "¿Cuál es el límite máximo para taxis?",
        "context": (
            "Política de gastos 2026. El uso de taxi puede reembolsarse si existe "
            "justificación, recibo válido y aprobación del responsable. El documento "
            "no indica un límite máximo para taxis."
        ),
        "ground_truth": (
            "No hay información suficiente para indicar un límite máximo. "
            "Solo consta que se exige justificación, recibo válido y aprobación."
        ),
        "expected_sources": ["POL-GASTOS-2026"],
    },
    {
        "id": "password_reset",
        "category": "soporte",
        "query": "¿Cómo restablezco mi contraseña corporativa?",
        "context": (
            "Manual de identidad. El usuario debe acceder al portal de autoservicio, "
            "validar MFA y completar el flujo de cambio de contraseña. Soporte L1 no debe "
            "pedir ni recibir contraseñas del usuario."
        ),
        "ground_truth": (
            "Debe usarse el portal de autoservicio, validando MFA. Soporte L1 no debe "
            "solicitar ni recibir la contraseña del usuario."
        ),
        "expected_sources": ["MANUAL-IDENTIDAD"],
    },
]


SYSTEM_PROMPT = """
Eres un agente de soporte L1 para un entorno enterprise.

Reglas:
- Responde solo con la información proporcionada en el contexto.
- Si el contexto no contiene la respuesta, dilo claramente.
- No inventes procedimientos, límites, importes ni acciones.
- Da una respuesta útil y accionable.
- Responde en español.
"""


@dataclass
class QualityCaseResult:
    case_id: str
    query: str
    response: str
    consistency_score: float
    consistency_passed: bool
    evaluator_results: dict[str, Any]
    passed: bool


def build_openai_client() -> AzureOpenAI:
    endpoint = os.environ["AOAI_ENDPOINT_PRIMARY"]
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
    api_key = os.getenv("AOAI_API_KEY_PRIMARY")

    if api_key:
        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )

    return AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version=api_version,
    )


def build_model_config() -> AzureOpenAIModelConfiguration:
    endpoint = os.environ["AOAI_ENDPOINT_PRIMARY"]
    deployment = os.getenv("AOAI_DEPLOYMENT_CHAT_PRIMARY", "gpt-4o")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
    api_key = os.getenv("AOAI_API_KEY_PRIMARY")

    if api_key:
        return AzureOpenAIModelConfiguration(
            azure_endpoint=endpoint,
            api_key=api_key,
            azure_deployment=deployment,
            api_version=api_version,
        )

    return AzureOpenAIModelConfiguration(
        azure_endpoint=endpoint,
        credential=DefaultAzureCredential(
            exclude_interactive_browser_credential=False
        ),
        azure_deployment=deployment,
        api_version=api_version,
    )


def call_agent(client: AzureOpenAI, query: str, context: str) -> str:
    deployment = os.getenv("AOAI_DEPLOYMENT_CHAT_PRIMARY", "gpt-4o")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Contexto:\n{context}\n\n"
                f"Pregunta del usuario:\n{query}\n\n"
                "Respuesta:"
            ),
        },
    ]

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=messages,
            temperature=0.2,
            max_completion_tokens=350,
        )
    except OpenAIError as ex:
        if "max_completion_tokens" not in str(ex):
            raise

        response = client.chat.completions.create(
            model=deployment,
            messages=messages,
            temperature=0.2,
            max_tokens=350,
        )

    return response.choices[0].message.content or ""


def normalized_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def consistency_score(responses: list[str]) -> float:
    if len(responses) < 2:
        return 1.0

    pair_scores = []

    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            pair_scores.append(normalized_similarity(responses[i], responses[j]))

    return statistics.mean(pair_scores)


def run_evaluators(
    model_config: AzureOpenAIModelConfiguration,
    query: str,
    context: str,
    response: str,
    ground_truth: str,
) -> dict[str, Any]:
    evaluators = {
        "groundedness": GroundednessEvaluator(model_config=model_config, threshold=3),
        "relevance": RelevanceEvaluator(model_config=model_config, threshold=3),
        "coherence": CoherenceEvaluator(model_config=model_config, threshold=3),
        "fluency": FluencyEvaluator(model_config=model_config, threshold=3),
        "similarity": SimilarityEvaluator(model_config=model_config, threshold=3),
    }

    results: dict[str, Any] = {}

    results["groundedness"] = evaluators["groundedness"](
        query=query,
        context=context,
        response=response,
    )

    results["relevance"] = evaluators["relevance"](
        query=query,
        response=response,
    )

    results["coherence"] = evaluators["coherence"](
        query=query,
        response=response,
    )

    results["fluency"] = evaluators["fluency"](
        query=query,
        response=response,
    )

    results["similarity"] = evaluators["similarity"](
        query=query,
        response=response,
        ground_truth=ground_truth,
    )

    return results


def evaluator_passed(result: dict[str, Any]) -> bool:
    """
    Normaliza distintos formatos de salida posibles del SDK.
    Busca campos habituales como *_result, label, passed o score.
    """
    text = json.dumps(result, ensure_ascii=False).lower()

    if '"fail"' in text or '"failed"' in text or '"passed": false' in text:
        return False

    return True


def evaluate_case(
    client: AzureOpenAI,
    model_config: AzureOpenAIModelConfiguration,
    case: dict[str, Any],
    repeats: int = 3,
) -> QualityCaseResult:
    responses = [
        call_agent(client, case["query"], case["context"])
        for _ in range(repeats)
    ]

    response = responses[0]

    consistency = consistency_score(responses)
    consistency_passed = consistency >= 0.80

    evaluator_results = run_evaluators(
        model_config=model_config,
        query=case["query"],
        context=case["context"],
        response=response,
        ground_truth=case["ground_truth"],
    )

    all_eval_passed = all(
        evaluator_passed(value)
        for value in evaluator_results.values()
    )

    passed = consistency_passed and all_eval_passed

    return QualityCaseResult(
        case_id=case["id"],
        query=case["query"],
        response=response,
        consistency_score=round(consistency, 3),
        consistency_passed=consistency_passed,
        evaluator_results=evaluator_results,
        passed=passed,
    )


def main() -> None:
    client = build_openai_client()
    model_config = build_model_config()

    results = [
        evaluate_case(client, model_config, case)
        for case in DATASET
    ]

    print("\n=== RESULTADOS DE EVALUACIÓN DE CALIDAD ===\n")

    failed = 0

    for result in results:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        print("-" * 100)

        if not result.passed:
            failed += 1

    total = len(results)
    defect_rate = failed / total if total else 0

    print("\n=== RESUMEN ===")
    print(f"Casos evaluados: {total}")
    print(f"Fallos: {failed}")
    print(f"Defect rate: {defect_rate:.2%}")

    if failed:
        raise SystemExit("La evaluación de calidad NO supera el umbral mínimo.")

    print("La evaluación de calidad supera el umbral mínimo.")


if __name__ == "__main__":
    main()