import os
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-10-21",
    azure_ad_token_provider=token_provider,
)

DEPLOYMENTS = {
    "chat": "gpt-4o",
    "classifier": "gpt-4o-mini",
    "embeddings": "text-embedding-3-small"
}


def choose_deployment(task_type: str) -> str:
    if task_type == "classification":
        return DEPLOYMENTS["classifier"]
    elif task_type == "chat":
        return DEPLOYMENTS["chat"]
    else:
        raise ValueError(f"Tipo de tarea no soportado: {task_type}")


def run_task(task_type: str, user_input: str) -> str:
    deployment = choose_deployment(task_type)

    if task_type == "classification":
        messages = [
            {"role": "system", "content": "Clasifica la consulta en una sola etiqueta: soporte, ventas o facturación."},
            {"role": "user", "content": user_input}
        ]
    else:
        messages = [
            {"role": "system", "content": "Eres un asistente empresarial."},
            {"role": "user", "content": user_input}
        ]

    response = client.chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=0.1 if task_type == "classification" else 0.3,
        max_tokens=50 if task_type == "classification" else 300
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    print(run_task("classification", "Necesito ayuda con una factura incorrecta"))
    print(run_task("chat", "Resume el papel de Azure OpenAI en una arquitectura enterprise"))