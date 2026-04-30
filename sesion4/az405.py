from azure.ai.evaluation import GroundednessEvaluator, AzureOpenAIModelConfiguration
import os

model_config = AzureOpenAIModelConfiguration(
    azure_endpoint=os.environ["AOAI_ENDPOINT_PRIMARY"],
    azure_deployment=os.environ["AOAI_DEPLOYMENT_CHAT_PRIMARY"],
    api_version="2024-10-21",
    api_key=os.environ["AOAI_API_KEY_PRIMARY"],
)

groundedness = GroundednessEvaluator(
    model_config=model_config,
    threshold=3,
)

result = groundedness(
    query="¿Puedo reclamar un taxi si vuelvo tarde de una visita a cliente?",
    context="""
    Los gastos de transporte derivados de visitas a cliente podrán ser reembolsados
    cuando estén justificados y aprobados por el responsable directo. El uso de taxi
    requiere justificación del motivo y recibo válido.
    """,
    response="""
    Sí, puedes reclamar un taxi si está justificado, aprobado por tu responsable
    y aportas un recibo válido.
    """,
)

print(result)