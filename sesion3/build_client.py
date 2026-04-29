import os
from typing import Optional

from openai import AzureOpenAI, OpenAI, OpenAIError
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.core.exceptions import AzureError, ClientAuthenticationError

from agent_framework.openai import OpenAIChatCompletionClient, OpenAIChatClient


DEFAULT_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")


def _build_azure_openai_v1_base_url(endpoint: str) -> str:
    """
    Normaliza el endpoint de Azure OpenAI para usarlo con el cliente OpenAI v1.

    Entrada esperada:
      https://<resource>.openai.azure.com/

    Salida:
      https://<resource>.openai.azure.com/openai/v1/
    """
    endpoint = (endpoint or "").strip().rstrip("/")

    if not endpoint:
        raise ValueError(
            "endpoint no puede estar vacío. "
            "Ejemplo: https://<recurso>.openai.azure.com/"
        )

    lower_endpoint = endpoint.lower()

    if lower_endpoint.endswith("/openai/v1"):
        return endpoint + "/"

    if lower_endpoint.endswith("/openai"):
        return endpoint + "/v1/"

    return endpoint + "/openai/v1/"


def build_client_chat(
    endpoint: str,
    api_key: Optional[str] = None,
    api_version: str = DEFAULT_API_VERSION,
) -> AzureOpenAI:
    """
    Cliente para Chat Completions clásico con AzureOpenAI.

    - Si api_key viene rellena, usa API Key.
    - Si api_key está vacía, usa Microsoft Entra ID / RBAC.
    """

    endpoint = (endpoint or "").strip()
    api_key = (api_key or "").strip()
    api_version = (api_version or "").strip()

    if not endpoint:
        raise ValueError(
            "endpoint no puede estar vacío. "
            "Ejemplo: https://<recurso>.openai.azure.com/"
        )

    if not api_version:
        raise ValueError(
            "api_version no puede estar vacía. "
            "Ejemplo: 2024-10-21"
        )

    try:
        if api_key:
            return AzureOpenAI(
                api_key=api_key,
                azure_endpoint=endpoint,
                api_version=api_version,
            )

        credential = DefaultAzureCredential()

        # Validación temprana para fallar aquí y no en la primera llamada al modelo.
        credential.get_token("https://cognitiveservices.azure.com/.default")

        token_provider = get_bearer_token_provider(
            credential,
            "https://cognitiveservices.azure.com/.default",
        )

        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_version=api_version,
            azure_ad_token_provider=token_provider,
        )

    except ClientAuthenticationError as exc:
        raise RuntimeError(
            "No se pudo autenticar con Microsoft Entra ID/RBAC para Chat Completions. "
            "Ejecuta `az login` o configura Managed Identity/Service Principal, "
            "y confirma que la identidad tiene `Cognitive Services OpenAI User` "
            "sobre el recurso Azure OpenAI."
        ) from exc

    except AzureError as exc:
        raise RuntimeError(
            "Error inicializando credenciales de Azure para Chat Completions."
        ) from exc

    except OpenAIError as exc:
        raise RuntimeError(
            "Error creando el cliente AzureOpenAI. "
            "Revisa endpoint, api_version y credenciales."
        ) from exc


def build_client_resp(
    endpoint: str,
    api_key: Optional[str] = None,
) -> OpenAI:
    """
    Cliente para Responses API con Azure OpenAI v1.

    - Si api_key viene rellena, usa API Key.
    - Si api_key está vacía, usa Microsoft Entra ID / RBAC.
    - Devuelve un cliente OpenAI.

    Requiere para RBAC:
    - `az login`, Managed Identity o Service Principal configurado.
    - Rol `Cognitive Services OpenAI User` sobre el recurso Azure OpenAI.
    """

    api_key = (api_key or "").strip()
    base_url = _build_azure_openai_v1_base_url(endpoint)

    try:
        if api_key:
            return OpenAI(
                api_key=api_key,
                base_url=base_url,
            )

        credential = DefaultAzureCredential()
        credential.get_token("https://ai.azure.com/.default")

        token_provider = get_bearer_token_provider(
            credential,
            "https://ai.azure.com/.default",
        )

        return OpenAI(
            api_key=token_provider,
            base_url=base_url,
        )

    except ClientAuthenticationError as exc:
        raise RuntimeError(
            "No se pudo autenticar con Microsoft Entra ID/RBAC para Responses API. "
            "Ejecuta `az login` o configura Managed Identity/Service Principal, "
            "y confirma que la identidad tiene `Cognitive Services OpenAI User` "
            "sobre el recurso Azure OpenAI."
        ) from exc

    except AzureError as exc:
        raise RuntimeError(
            "Error inicializando credenciales de Azure para Responses API."
        ) from exc

    except OpenAIError as exc:
        raise RuntimeError(
            "Error creando el cliente OpenAI para Responses API. "
            "Revisa endpoint, base_url y credenciales."
        ) from exc
    

def build_maf_chat_client(
    endpoint: str,
    model: str,
    api_key: Optional[str] = None,
    api_version: str = DEFAULT_API_VERSION,
):
    """
    Crea un OpenAIChatCompletionClient para Microsoft Agent Framework.

    - Si api_key viene informada, usa API Key.
    - Si api_key está vacía, usa Microsoft Entra ID / RBAC.
    - Devuelve (client, credential).
      Si se usa API Key, credential será None.
      Si se usa RBAC, debes cerrar credential con await credential.close().
    """

    if not endpoint:
        raise ValueError("Falta endpoint. Define AZURE_OPENAI_ENDPOINT.")

    if not model:
        raise ValueError("Falta model/deployment. Define AZURE_OPENAI_DEPLOYMENT_1.")

    if not api_version:
        raise ValueError("Falta api_version. Define AZURE_OPENAI_API_VERSION.")

    if api_key:
        client = OpenAIChatCompletionClient(
            azure_endpoint=endpoint,
            model=model,
            api_version=api_version,
            api_key=api_key,
        )

        return client

    credential = DefaultAzureCredential(
        exclude_interactive_browser_credential=False
    )

    client = OpenAIChatCompletionClient(
        azure_endpoint=endpoint,
        model=model,
        api_version=api_version,
        credential=credential,
    )

    return client

def build_maf_responses_client(
    endpoint: str,
    model: str,
    api_key: Optional[str] = None,
    api_version: str = DEFAULT_API_VERSION,
):
    """
    Cliente MAF para Responses API.

    Usa:
    - API Key si api_key viene informada.
    - RBAC / Microsoft Entra ID si api_key está vacía.

    Devuelve:
    - client
    - credential, o None si se usa API Key.

    Nota:
    - model debe ser el nombre del deployment en Azure.
    - endpoint puede ser un Azure OpenAI endpoint:
        https://<resource>.openai.azure.com/
      o, si el SDK/cliente lo soporta en tu versión, un endpoint de proyecto Foundry.
    """

    if not endpoint:
        raise ValueError("Falta endpoint.")

    if not model:
        raise ValueError("Falta model/deployment.")

    if not api_version:
        raise ValueError("Falta api_version.")

    if api_key:
        client = OpenAIChatClient(
            azure_endpoint=endpoint,
            model=model,
            api_version=api_version,
            api_key=api_key,
        )
        return client

    credential = DefaultAzureCredential(
        exclude_interactive_browser_credential=False
    )

    client = OpenAIChatClient(
        azure_endpoint=endpoint,
        model=model,
        api_version=api_version,
        credential=credential,
    )

    return client