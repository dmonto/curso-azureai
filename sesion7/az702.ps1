# foundry_inventory.ps1
# Inventario básico de Microsoft Foundry / Azure AI:
# - Recursos Foundry / Azure OpenAI
# - Proyectos Foundry
# - Deployments del recurso
# - Agentes y deployments vistos desde el proyecto Foundry
Write-Host "=== SCRIPT INICIADO ===" -ForegroundColor Green
Write-Host "Ruta actual: $(Get-Location)"
Write-Host "Usuario: $env:USERNAME"
Write-Host "PowerShell: $($PSVersionTable.PSVersion)"
Read-Host "Pulsa ENTER para continuar"
$ErrorActionPreference = "Stop"

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

$env:SUBSCRIPTION_ID = "1987bab9-33e4-49d8-952a-c9034c49b8bf"
$env:RESOURCE_GROUP = "Curso-IA"

# Foundry resource / AI Services account
$env:FOUNDRY_RESOURCE = "curso-ia-foundry"

# Foundry project
$env:FOUNDRY_PROJECT = "curso-ia-proj"

# Azure OpenAI resource, si lo tenéis separado
$env:AZURE_OPENAI_RESOURCE = "curso-ia-openai-responses"

# Endpoint del proyecto Foundry.
# Formato:
# https://<resource-name>.services.ai.azure.com/api/projects/<project-name>
$env:FOUNDRY_PROJECT_ENDPOINT = "https://curso-ia-openai-responses.services.ai.azure.com/api/projects/responses-project"


# ============================================================
# 2. LOGIN Y CONTEXTO
# ============================================================

Write-Host "`n== Azure CLI context ==" -ForegroundColor Cyan

az account show 1>$null 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "No hay sesión activa. Ejecutando az login..." -ForegroundColor Yellow
    az login
}

az account set --subscription $env:SUBSCRIPTION_ID

Write-Host "Subscription:" $env:SUBSCRIPTION_ID
Write-Host "Resource group:" $env:RESOURCE_GROUP


# ============================================================
# 4. LISTAR RECURSOS COGNITIVE SERVICES / AI SERVICES / OPENAI
# ============================================================

Write-Host "`n== Recursos Azure AI / Foundry / OpenAI en el resource group ==" -ForegroundColor Cyan

az cognitiveservices account list `
    --resource-group $env:RESOURCE_GROUP `
    --query "[].{name:name, kind:kind, location:location, sku:sku.name, endpoint:properties.endpoint}" `
    -o table


# ============================================================
# 5. MOSTRAR DETALLE DEL RECURSO FOUNDRY
# ============================================================

Write-Host "`n== Detalle Foundry resource ==" -ForegroundColor Cyan

az cognitiveservices account show `
    --name $env:FOUNDRY_RESOURCE `
    --resource-group $env:RESOURCE_GROUP `
    --query "{name:name, kind:kind, location:location, endpoint:properties.endpoint, customSubDomainName:properties.customSubDomainName}" `
    -o jsonc


# ============================================================
# 6. LISTAR PROYECTOS DEL RECURSO FOUNDRY
# ============================================================

Write-Host "`n== Proyectos Foundry bajo el recurso ==" -ForegroundColor Cyan

az cognitiveservices account project list `
    --name $env:FOUNDRY_RESOURCE `
    --resource-group $env:RESOURCE_GROUP `
    --query "[].{name:name, location:location, provisioningState:properties.provisioningState}" `
    -o table


# ============================================================
# 7. MOSTRAR DETALLE DEL PROYECTO
# ============================================================

Write-Host "`n== Detalle del proyecto Foundry ==" -ForegroundColor Cyan

az cognitiveservices account project show `
    --name $env:FOUNDRY_RESOURCE `
    --resource-group $env:RESOURCE_GROUP `
    --project-name $env:FOUNDRY_PROJECT `
    -o jsonc


# ============================================================
# 8. LISTAR CONEXIONES DEL PROYECTO
# ============================================================

Write-Host "`n== Conexiones del proyecto Foundry ==" -ForegroundColor Cyan

az cognitiveservices account project connection list `
    --name $env:FOUNDRY_RESOURCE `
    --resource-group $env:RESOURCE_GROUP `
    --project-name $env:FOUNDRY_PROJECT `
    --query "[].{name:name, type:properties.category, target:properties.target}" `
    -o table


# ============================================================
# 9. LISTAR DEPLOYMENTS DEL RECURSO FOUNDRY
# ============================================================

Write-Host "`n== Deployments del recurso Foundry ==" -ForegroundColor Cyan

az cognitiveservices account deployment list `
    --name $env:FOUNDRY_RESOURCE `
    --resource-group $env:RESOURCE_GROUP `
    --query "[].{name:name, model:properties.model.name, version:properties.model.version, format:properties.model.format, sku:sku.name, capacity:sku.capacity}" `
    -o table


# ============================================================
# 10. LISTAR DEPLOYMENTS DEL RECURSO AZURE OPENAI, SI EXISTE
# ============================================================

Write-Host "`n== Deployments del recurso Azure OpenAI ==" -ForegroundColor Cyan

try {
    az cognitiveservices account deployment list `
        --name $env:AZURE_OPENAI_RESOURCE `
        --resource-group $env:RESOURCE_GROUP `
        --query "[].{name:name, model:properties.model.name, version:properties.model.version, format:properties.model.format, sku:sku.name, capacity:sku.capacity}" `
        -o table
}
catch {
    Write-Host "No se pudieron listar deployments de Azure OpenAI. Revisa nombre del recurso o permisos." -ForegroundColor Yellow
}


# ============================================================
# 11. LISTAR AGENTES Y DEPLOYMENTS DESDE EL PROYECTO FOUNDRY
#     usando azure-ai-projects
# ============================================================

Write-Host "`n== Agentes y deployments vía Foundry SDK ==" -ForegroundColor Cyan

$pythonScript = @'
import os
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient


endpoint = os.environ["FOUNDRY_PROJECT_ENDPOINT"]

client = AIProjectClient(
    endpoint=endpoint,
    credential=DefaultAzureCredential(exclude_interactive_browser_credential=False),
)

print("\n-- Foundry project endpoint --")
print(endpoint)

print("\n-- Agents --")
try:
    agents = list(client.agents.list())
    if not agents:
        print("(sin agentes)")
    for agent in agents:
        print("-" * 80)
        print("name:", getattr(agent, "name", None) or agent.get("name"))
        print("id:", getattr(agent, "id", None) or agent.get("id"))
        print("kind:", getattr(agent, "kind", None) or agent.get("kind"))
        print("created_at:", getattr(agent, "created_at", None) or agent.get("created_at"))

        agent_name = getattr(agent, "name", None) or agent.get("name")
        if agent_name:
            print("versions:")
            try:
                for version in client.agents.list_versions(agent_name=agent_name):
                    print("  - version:", getattr(version, "version", None) or version.get("version"),
                          "| status:", getattr(version, "status", None) or version.get("status"))
            except Exception as version_error:
                print("  no se pudieron listar versiones:", version_error)
except Exception as agent_error:
    print("No se pudieron listar agentes:", agent_error)

print("\n-- Project deployments --")
try:
    deployments = list(client.deployments.list())
    if not deployments:
        print("(sin deployments visibles desde el proyecto)")
    for deployment in deployments:
        print("-" * 80)
        print(deployment)
except Exception as deployment_error:
    print("No se pudieron listar deployments del proyecto:", deployment_error)
'@

$tempFile = Join-Path $env:TEMP "foundry_inventory_sdk.py"
$pythonScript | Out-File -FilePath $tempFile -Encoding utf8

python -m pip install --quiet --upgrade azure-ai-projects azure-identity
python $tempFile

Write-Host "`nInventario finalizado." -ForegroundColor Green