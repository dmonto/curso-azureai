

import os
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda


load_dotenv()


ExpenseType = Literal["hotel", "taxi", "comida", "otro"]
Decision = Literal["APROBABLE", "REQUIERE_JUSTIFICACION", "FUERA_DE_POLITICA"]


class ExpenseExtraction(BaseModel):
    expense_type: ExpenseType = Field(
        description="Tipo de gasto detectado: hotel, taxi, comida u otro"
    )
    amount_eur: Optional[float] = Field(
        default=None,
        description="Importe principal detectado en euros"
    )
    unit: Optional[str] = Field(
        default=None,
        description="Unidad del importe: noche, trayecto, persona, total o desconocido"
    )
    quantity: Optional[int] = Field(
        default=None,
        description="Cantidad asociada si aparece: noches, personas, trayectos, etc."
    )
    destination: Optional[str] = Field(
        default=None,
        description="Ciudad o destino si aparece"
    )
    business_reason: Optional[str] = Field(
        default=None,
        description="Motivo profesional del gasto si aparece"
    )
    missing_data: list[str] = Field(
        default_factory=list,
        description="Datos importantes que faltan para decidir con seguridad"
    )


class PolicyEvaluation(BaseModel):
    decision: Decision
    reason: str
    next_step: str
    extracted: ExpenseExtraction


def build_model() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint=os.environ["AOAI_ENDPOINT_SECONDARY"],
        api_key=os.environ["AOAI_API_KEY_SECONDARY"],
        azure_deployment=os.environ["AOAI_DEPLOYMENT_CHAT_SECONDARY"],
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        temperature=0.1,
    )


def evaluate_policy(extracted: ExpenseExtraction) -> PolicyEvaluation:
    """
    Aplica reglas de negocio de forma determinista.
    Aquí no decide el LLM.
    """

    if extracted.missing_data:
        return PolicyEvaluation(
            decision="REQUIERE_JUSTIFICACION",
            reason=(
                "Faltan datos relevantes para validar la solicitud: "
                + ", ".join(extracted.missing_data)
            ),
            next_step="Solicitar información adicional al empleado.",
            extracted=extracted,
        )

    if extracted.amount_eur is None:
        return PolicyEvaluation(
            decision="REQUIERE_JUSTIFICACION",
            reason="No se ha identificado un importe claro en la solicitud.",
            next_step="Pedir al empleado que indique el importe estimado.",
            extracted=extracted,
        )

    amount = extracted.amount_eur

    if extracted.expense_type == "hotel":
        if amount <= 150:
            decision = "APROBABLE"
            reason = "El importe del hotel está dentro del límite de 150€/noche."
            next_step = "Registrar la solicitud como preaprobada."
        elif amount <= 220:
            decision = "REQUIERE_JUSTIFICACION"
            reason = "El hotel supera el límite estándar de 150€/noche, pero no excede el máximo excepcional de 220€/noche."
            next_step = "Solicitar justificación o aprobación del responsable."
        else:
            decision = "FUERA_DE_POLITICA"
            reason = "El hotel supera el máximo permitido de 220€/noche."
            next_step = "Rechazar la solicitud o escalarla a revisión excepcional."

    elif extracted.expense_type == "taxi":
        if amount <= 40:
            decision = "APROBABLE"
            reason = "El importe del taxi está dentro del límite de 40€."
            next_step = "Registrar la solicitud como preaprobada."
        else:
            decision = "REQUIERE_JUSTIFICACION"
            reason = "El taxi supera el límite estándar de 40€."
            next_step = "Solicitar justificación del trayecto."

    elif extracted.expense_type == "comida":
        if amount <= 35:
            decision = "APROBABLE"
            reason = "El importe de comida está dentro del límite de 35€ por persona."
            next_step = "Registrar la solicitud como preaprobada."
        else:
            decision = "REQUIERE_JUSTIFICACION"
            reason = "La comida supera el límite de 35€ por persona."
            next_step = "Solicitar justificación o aprobación del responsable."

    else:
        decision = "REQUIERE_JUSTIFICACION"
        reason = "El tipo de gasto no encaja claramente en hotel, taxi o comida."
        next_step = "Revisar manualmente la solicitud."

    return PolicyEvaluation(
        decision=decision,
        reason=reason,
        next_step=next_step,
        extracted=extracted,
    )


def build_expense_chain():
    model = build_model()

    extraction_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Eres un extractor de datos para solicitudes de gastos corporativos.

Extrae información estructurada de la solicitud.
No apruebes ni rechaces nada.
No apliques la política.
Solo extrae datos.

Reglas:
- Si es hotel, intenta extraer importe por noche.
- Si es taxi, intenta extraer importe del trayecto.
- Si es comida, intenta extraer importe por persona.
- Si falta importe, añade "importe" a missing_data.
- Si falta motivo profesional, añade "motivo profesional" a missing_data.
""",
            ),
            (
                "user",
                "Solicitud: {request}",
            ),
        ]
    )

    extraction_chain = extraction_prompt | model.with_structured_output(ExpenseExtraction)

    policy_chain = RunnableLambda(evaluate_policy)

    response_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Eres un asistente corporativo de gastos.

Redacta una respuesta breve y profesional para el empleado.
No cambies la decisión recibida.
No añadas reglas nuevas.
No inventes datos.
""",
            ),
            (
                "user",
                """
Solicitud original:
{request}

Evaluación:
Decisión: {decision}
Motivo: {reason}
Siguiente paso: {next_step}

Datos extraídos:
{extracted}
""",
            ),
        ]
    )

    response_chain = response_prompt | model

    def run_chain(inputs: dict) -> str:
        request = inputs["request"]

        extracted: ExpenseExtraction = extraction_chain.invoke({"request": request})
        evaluation: PolicyEvaluation = policy_chain.invoke(extracted)

        extracted_json = evaluation.extracted.model_dump_json(indent=2)

        final_response = response_chain.invoke(
            {
                "request": request,
                "decision": evaluation.decision,
                "reason": evaluation.reason,
                "next_step": evaluation.next_step,
                "extracted": extracted_json,
            }
        )

        return final_response.content

    return RunnableLambda(run_chain)


def check_travel_expense(request: str) -> str:
    chain = build_expense_chain()
    return chain.invoke({"request": request})


if __name__ == "__main__":
    examples = [
        "Quiero reservar un hotel de 180€ por noche en Madrid para asistir a una reunión con cliente durante 2 noches.",
        "Necesito un taxi de 55€ desde el aeropuerto hasta el hotel por una visita comercial.",
        "Tengo una comida de equipo de 28€ por persona con un cliente.",
        "Quiero reservar un hotel para una conferencia.",
    ]

    for request in examples:
        print("=" * 80)
        print("SOLICITUD:")
        print(request)
        print("\nRESPUESTA:")
        print(check_travel_expense(request))    