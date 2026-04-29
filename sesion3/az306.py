import os

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Falta la variable de entorno {name}")
    return value


def build_secret_client() -> SecretClient:
    vault_name = require_env("AZURE_KEYVAULT_NAME")
    vault_url = f"https://{vault_name}.vault.azure.net"

    credential = DefaultAzureCredential(
        exclude_interactive_browser_credential=False
    )

    return SecretClient(
        vault_url=vault_url,
        credential=credential,
    )


def main() -> None:
    secret_name = require_env("AZURE_KEYVAULT_SECRET_NAME")

    client = build_secret_client()

    secret = client.get_secret(secret_name)

    print("Secreto recuperado correctamente")
    print(f"Nombre: {secret.name}")
    print(f"Valor: {secret.value}")


if __name__ == "__main__":
    main()