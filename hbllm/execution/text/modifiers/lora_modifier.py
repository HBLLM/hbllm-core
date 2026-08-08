"""
LoRA Modifier — wraps existing LoRAManager behind the modifier interface.

This is the bridge between the Execution OS and the existing
LoRA infrastructure in ``hbllm.modules.lora``. The cognitive layer
never imports LoRA directly — it flows through here.

The modifier activates/deactivates LoRA adapters during the
``before_generation`` / ``cleanup`` lifecycle hooks.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.execution.plan import ExecutionPlan

logger = logging.getLogger(__name__)


class LoRAModifier:
    """
    Wraps existing LoRAManager/LoRALinear behind the GenerationModifier interface.

    Lifecycle:
        before_generation → Sets ACTIVE_ADAPTER context var
        cleanup → Resets ACTIVE_ADAPTER

    The existing LoRALinear.forward() reads ACTIVE_ADAPTER
    automatically — no changes needed to the low-level LoRA code.
    """

    def __init__(
        self,
        adapter_name: str = "default",
        model: Any = None,  # torch.nn.Module | None
        lora_r: int = 8,
        lora_alpha: float = 16.0,
    ) -> None:
        self._adapter_name = adapter_name
        self._model = model
        self._lora_r = lora_r
        self._lora_alpha = lora_alpha
        self._active = False
        self._token: Any = None  # contextvars.Token

    @property
    def name(self) -> str:
        return f"lora-{self._adapter_name}"

    @property
    def modifier_type(self) -> str:
        return "lora"

    def priority(self) -> int:
        return 100  # LoRA should activate early

    def supports(self, model_id: str) -> bool:
        # LoRA only works with local models
        return self._model is not None

    async def is_available(self) -> bool:
        """LoRA is available if a model is loaded and adapter weights exist."""
        if self._model is None:
            return False
        try:
            from hbllm.modules.lora import LoRALinear

            # Check if any LoRA layers have this adapter
            for module in self._model.modules():
                if isinstance(module, LoRALinear) and self._adapter_name in module.lora_A:
                    return True
        except ImportError:
            pass
        return False

    async def before_context(self, plan: ExecutionPlan) -> ExecutionPlan:
        return plan

    async def before_prompt(self, prompt: str, plan: ExecutionPlan) -> str:
        return prompt

    async def before_generation(self, plan: ExecutionPlan) -> None:
        """Activate the LoRA adapter by setting the context variable."""
        try:
            from hbllm.modules.lora import ACTIVE_ADAPTER

            self._token = ACTIVE_ADAPTER.set(self._adapter_name)
            self._active = True
            logger.debug("LoRA adapter activated: %s", self._adapter_name)
        except ImportError:
            logger.warning("LoRA module not available, skipping activation")

    async def after_generation(self, text: str, plan: ExecutionPlan) -> str:
        return text

    async def after_validation(self, text: str, plan: ExecutionPlan) -> str:
        return text

    async def cleanup(self) -> None:
        """Deactivate the LoRA adapter by resetting the context variable."""
        if self._active and self._token is not None:
            try:
                from hbllm.modules.lora import ACTIVE_ADAPTER

                ACTIVE_ADAPTER.reset(self._token)
                self._active = False
                self._token = None
                logger.debug("LoRA adapter deactivated: %s", self._adapter_name)
            except ImportError:
                pass

    def metrics(self) -> dict[str, Any]:
        return {
            "type": "lora",
            "adapter_name": self._adapter_name,
            "active": self._active,
            "lora_r": self._lora_r,
            "lora_alpha": self._lora_alpha,
        }
