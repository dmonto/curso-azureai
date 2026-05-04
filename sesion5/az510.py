import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    AzureAIAgentTarget,
    AgentTaxonomyInput,
    EvaluationTaxonomy,
    RiskCategory,
)

load_dotenv()


def to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "as_dict"):
        return obj.as_dict()
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, list):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    return obj


def main() -> None:
    endpoint = os.environ["FOUNDRY_PROJECT"]
    agent_name = os.environ["AZURE_AI_AGENT_NAME"]
    agent_version = os.getenv("AZURE_AI_AGENT_VERSION", "1")
    model_deployment = os.environ["AOAI_DEPLOYMENT_CHAT_PRIMARY"]

    credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)

    with AIProjectClient(endpoint=endpoint, credential=credential) as project_client:
        openai_client = project_client.get_openai_client()

        # 1. Target: tu agente Foundry
        target = AzureAIAgentTarget(
            name=agent_name,
            version=agent_version,
        )

        print(f"Target agent: {agent_name} v{agent_version}")

        # 2. Crear grupo de evaluación Red Team
        red_team = openai_client.evals.create(
            name=f"Red Team - {agent_name}",
            data_source_config={
                "type": "azure_ai_source",
                "scenario": "red_team",
            },
            testing_criteria=[
                {
                    "type": "azure_ai_evaluator",
                    "name": "Prohibited Actions",
                    "evaluator_name": "builtin.prohibited_actions",
                    "evaluator_version": "1",
                },
                {
                    "type": "azure_ai_evaluator",
                    "name": "Task Adherence",
                    "evaluator_name": "builtin.task_adherence",
                    "evaluator_version": "1",
                    "initialization_parameters": {
                        "deployment_name": model_deployment,
                    },
                },
                {
                    "type": "azure_ai_evaluator",
                    "name": "Sensitive Data Leakage",
                    "evaluator_name": "builtin.sensitive_data_leakage",
                    "evaluator_version": "1",
                },
            ],
        )

        print(f"Red team eval creado: {red_team.id}")

        # 3. Crear taxonomy para Prohibited Actions
        taxonomy = project_client.beta.evaluation_taxonomies.create(
            name=f"{agent_name}-taxonomy",
            body=EvaluationTaxonomy(
                description="Taxonomy for AI red teaming run",
                taxonomy_input=AgentTaxonomyInput(
                    risk_categories=[RiskCategory.PROHIBITED_ACTIONS],
                    target=target,
                ),
            ),
        )

        taxonomy_file_id = taxonomy.id
        print(f"Taxonomy creada: {taxonomy_file_id}")

        # 4. Crear run
        eval_run = openai_client.evals.runs.create(
            eval_id=red_team.id,
            name=f"Red Team Run - {agent_name}",
            data_source={
                "type": "azure_ai_red_team",
                "item_generation_params": {
                    "type": "red_team_taxonomy",
                    "attack_strategies": [
                        "Flip",
                        "Base64",
                        "IndirectJailbreak",
                    ],
                    "num_turns": 3,
                    "source": {
                        "type": "file_id",
                        "id": taxonomy_file_id,
                    },
                },
                "target": target.as_dict(),
            },
        )

        print(f"Run creado: {eval_run.id}")
        print(f"Estado inicial: {eval_run.status}")

        # 5. Polling hasta finalizar
        while True:
            run = openai_client.evals.runs.retrieve(
                eval_id=red_team.id,
                run_id=eval_run.id,
            )

            print(f"Estado: {run.status}")

            if run.status in ("completed", "failed", "canceled"):
                break

            time.sleep(10)

        # 6. Descargar resultados
        items = list(
            openai_client.evals.runs.output_items.list(
                eval_id=red_team.id,
                run_id=eval_run.id,
            )
        )

        output_path = Path(f"redteam_results_{agent_name}.json")
        output_path.write_text(
            json.dumps(to_jsonable(items), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"Resultado final: {run.status}")
        print(f"Output items: {len(items)}")
        print(f"Guardado en: {output_path.resolve()}")


if __name__ == "__main__":
    main()