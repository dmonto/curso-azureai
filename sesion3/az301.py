import asyncio
from typing_extensions import Never

from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler


class ClassifierExecutor(Executor):
    @handler
    async def classify(self, text: str, ctx: WorkflowContext[str]) -> None:
        lower = text.lower()

        if "factura" in lower or "cobro" in lower:
            await ctx.send_message("billing")
        elif "error" in lower or "vpn" in lower or "acceso" in lower:
            await ctx.send_message("support")
        else:
            await ctx.send_message("general")


class BillingExecutor(Executor):
    @handler
    async def process(self, category: str, ctx: WorkflowContext[Never, str]) -> None:
        await ctx.yield_output(
            "Ruta: facturación. Crear caso financiero y pedir número de factura."
        )


class SupportExecutor(Executor):
    @handler
    async def process(self, category: str, ctx: WorkflowContext[Never, str]) -> None:
        await ctx.yield_output(
            "Ruta: soporte. Crear ticket técnico y pedir descripción del error."
        )


class GeneralExecutor(Executor):
    @handler
    async def process(self, category: str, ctx: WorkflowContext[Never, str]) -> None:
        await ctx.yield_output(
            "Ruta: general. Solicitar más información antes de escalar."
        )


def is_billing(category: str) -> bool:
    return category == "billing"


def is_support(category: str) -> bool:
    return category == "support"


def is_general(category: str) -> bool:
    return category == "general"


async def main() -> None:
    classifier = ClassifierExecutor(id="classifier")
    billing = BillingExecutor(id="billing")
    support = SupportExecutor(id="support")
    general = GeneralExecutor(id="general")

    workflow = (
        WorkflowBuilder(
            start_executor=classifier,
            name="SupportRoutingWorkflow",
            description="Clasifica una petición y la enruta a la rama adecuada.",
        )
        .add_edge(classifier, billing, condition=is_billing)
        .add_edge(classifier, support, condition=is_support)
        .add_edge(classifier, general, condition=is_general)
        .build()
    )

    events = await workflow.run(
        "Tengo un error al acceder a la VPN corporativa."
    )

    print(events.get_outputs())


if __name__ == "__main__":
    asyncio.run(main())