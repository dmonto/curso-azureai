import os
import sys

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_openai import AzureChatOpenAI


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Falta la variable de entorno {name}")
    return value


def main() -> None:
    endpoint = require_env("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_1", "gpt-4o")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

    credential = DefaultAzureCredential(
        exclude_interactive_browser_credential=False
    )

    token_provider = get_bearer_token_provider(
        credential,
        "https://cognitiveservices.azure.com/.default"
    )

    model = AzureChatOpenAI(
        azure_endpoint=endpoint,
        azure_deployment=deployment,
        api_version=api_version,
        azure_ad_token_provider=token_provider,
        temperature=0.2,
        max_tokens=200,
    )

    response = model.invoke(
        "Explica en dos frases qué aporta Foundry sobre LangGraph puro."
    )

    response.pretty_print()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ERROR ejecutando Azure OpenAI Chat Completions")
        print(str(exc))
        sys.exit(1)