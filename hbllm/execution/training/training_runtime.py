"""
Training Runtime — training as just another execution backend.

In the Execution OS architecture, training is not a special subsystem.
It's an execution backend like text generation, vision, or audio.

The TrainingRuntime:
  - Subscribes to ``skill.discovered`` and ``learning.feedback_queued`` events
  - Dispatches training jobs as ExecutionRequests through the ExecutionBus
  - Returns TrainingResult with adapter state, metrics, checkpoint path
  - Zero cognitive knowledge — it doesn't know about domains or styles

Data flow:
    SpawnerNode → skill.discovered event
        ↓
    TrainingRuntime (execution backend)
        ↓
    LoRA/DPO training → TrainingResult → ExecutionBus event

The cognitive layer never sees training details.
The training runtime never sees cognitive metadata.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from hbllm.execution.capability import RuntimeCapabilities
from hbllm.execution.plan import ExecutionPlan, TaskType
from hbllm.execution.result import ExecutionMetrics, ExecutionResult, TokenUsage

logger = logging.getLogger(__name__)


# ── Training-specific types ──────────────────────────────────────────────────


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for a training job.

    This is an execution-level configuration — no cognitive metadata.
    The cognitive layer decides WHAT to train (via skill.discovered).
    This config decides HOW to train.
    """

    rank: int = 8  # LoRA rank
    alpha: float = 16.0  # LoRA alpha
    learning_rate: float = 1e-4
    max_steps: int = 20
    batch_size: int = 2
    max_samples: int = 50
    epochs: int = 2
    gradient_clip: float = 1.0
    weight_decay: float = 0.01
    checkpoint_dir: str = "./checkpoints/domains"
    training_type: str = "sft"  # "sft" | "dpo"


@dataclass
class TrainingResult:
    """Result of a training job.

    Contains only execution-level information — no cognitive metadata.
    """

    adapter_name: str = ""
    checkpoint_path: str = ""
    steps_completed: int = 0
    final_loss: float = 0.0
    training_time_ms: float = 0.0
    adapter_state: dict[str, Any] | None = None
    status: str = "completed"  # "completed" | "failed" | "cancelled"
    error: str = ""
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingJob:
    """A training job tracked by the runtime."""

    job_id: str = ""
    adapter_name: str = ""
    config: TrainingConfig = field(default_factory=TrainingConfig)
    dataset_path: str = ""
    status: str = "pending"  # "pending" | "running" | "completed" | "failed"
    result: TrainingResult | None = None

    def __post_init__(self) -> None:
        if not self.job_id:
            self.job_id = f"train-{uuid.uuid4().hex[:12]}"


# ── Training Runtime ─────────────────────────────────────────────────────────


class TrainingRuntime:
    """
    Training as an execution backend.

    The TrainingRuntime executes training jobs dispatched through the
    Execution OS. It wraps the existing SFT/DPO training code but
    presents it through the standard runtime interface.

    Supported task types:
        - LORA_TRAINING: Train a new LoRA adapter (SFT)
        - DPO_TRAINING: Direct Preference Optimization training

    This runtime has zero cognitive knowledge. It receives:
        - An adapter name (string)
        - A dataset path or training pairs
        - A TrainingConfig

    It returns:
        - A TrainingResult with adapter state and metrics
    """

    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        default_config: TrainingConfig | None = None,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._config = default_config or TrainingConfig()
        self._active_jobs: dict[str, TrainingJob] = {}
        self._completed_jobs: list[TrainingJob] = []

    @property
    def runtime_type(self) -> str:
        return "training"

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            supports_lora=True,
        )

    def supported_task_types(self) -> list[TaskType]:
        return [TaskType.LORA_TRAINING, TaskType.DPO_TRAINING]

    async def is_available(self) -> bool:
        """Training is available if we have a model and tokenizer."""
        return self._model is not None and self._tokenizer is not None

    async def execute(self, plan: ExecutionPlan) -> ExecutionResult:
        """Execute a training plan.

        Dispatches to SFT or DPO training based on the plan's task_type.
        """
        start_time = time.monotonic()

        try:
            # Extract training parameters from plan
            adapter_name = plan.metadata.get("adapter_name", plan.plan_id)
            dataset_path = plan.metadata.get("dataset_path", "")
            config = self._resolve_config(plan)

            # Create tracking job
            job = TrainingJob(
                adapter_name=adapter_name,
                config=config,
                dataset_path=dataset_path,
                status="running",
            )
            self._active_jobs[job.job_id] = job

            # Dispatch to appropriate training method
            if plan.task_type == TaskType.DPO_TRAINING:
                training_pairs = plan.metadata.get("training_pairs", [])
                result = await self._train_dpo(adapter_name, training_pairs, config)
            else:
                result = await self._train_sft(adapter_name, dataset_path, config)

            # Complete the job
            job.status = result.status
            job.result = result
            self._active_jobs.pop(job.job_id, None)
            self._completed_jobs.append(job)

            latency_ms = (time.monotonic() - start_time) * 1000

            # Propagate training failure status
            if result.status == "failed":
                return ExecutionResult(
                    content=f"Training failed: {result.error}",
                    plan_id=plan.plan_id,
                    status="failed",
                    usage=TokenUsage(),
                    metrics=ExecutionMetrics(latency_ms=latency_ms),
                    metadata={"error": result.error, "adapter_name": adapter_name},
                )

            return ExecutionResult(
                content=f"Training completed: {adapter_name} ({result.steps_completed} steps, loss={result.final_loss:.4f})",
                plan_id=plan.plan_id,
                status="completed",
                usage=TokenUsage(),
                metrics=ExecutionMetrics(
                    latency_ms=latency_ms,
                    modifiers_applied=[],
                ),
                metadata={
                    "adapter_name": adapter_name,
                    "checkpoint_path": result.checkpoint_path,
                    "steps": result.steps_completed,
                    "final_loss": result.final_loss,
                    "training_time_ms": result.training_time_ms,
                },
            )

        except Exception as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            logger.error("Training execution failed: %s", e)
            return ExecutionResult(
                content=f"Training failed: {e}",
                plan_id=plan.plan_id,
                status="failed",
                usage=TokenUsage(),
                metrics=ExecutionMetrics(latency_ms=latency_ms),
                metadata={"error": str(e)},
            )

    def _resolve_config(self, plan: ExecutionPlan) -> TrainingConfig:
        """Resolve training config from plan metadata, falling back to defaults."""
        meta = plan.metadata
        return TrainingConfig(
            rank=meta.get("rank", self._config.rank),
            alpha=meta.get("alpha", self._config.alpha),
            learning_rate=meta.get("learning_rate", self._config.learning_rate),
            max_steps=meta.get("max_steps", self._config.max_steps),
            batch_size=meta.get("batch_size", self._config.batch_size),
            max_samples=meta.get("max_samples", self._config.max_samples),
            epochs=meta.get("epochs", self._config.epochs),
            training_type=meta.get("training_type", self._config.training_type),
            checkpoint_dir=meta.get("checkpoint_dir", self._config.checkpoint_dir),
        )

    async def _train_sft(
        self,
        adapter_name: str,
        dataset_path: str,
        config: TrainingConfig,
    ) -> TrainingResult:
        """Execute SFT training.

        Wraps the existing hbllm.training.sft pipeline but through
        the standard execution interface.
        """
        import asyncio

        start = time.monotonic()

        def _do_train() -> TrainingResult:
            try:
                import torch

                from hbllm.modules.lora import LoRAManager
                from hbllm.training.sft import InstructionDataset, collate_sft, load_sft_data

                # Load dataset
                raw_data = load_sft_data(dataset_path)
                if not raw_data:
                    return TrainingResult(
                        adapter_name=adapter_name,
                        status="failed",
                        error="No training data loaded",
                    )

                dataset = InstructionDataset(
                    raw_data[: config.max_samples],
                    self._tokenizer,
                    max_length=256,
                )
                if len(dataset) == 0:
                    return TrainingResult(
                        adapter_name=adapter_name,
                        status="failed",
                        error="Empty dataset after preparation",
                    )

                loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=config.batch_size,
                    shuffle=True,
                    collate_fn=collate_sft,
                )

                # Inject LoRA
                LoRAManager.inject(self._model, r=config.rank)
                LoRAManager.add_adapter(self._model, adapter_name)
                LoRAManager.set_active_adapter(self._model, adapter_name)

                # Optimizer targeting only LoRA parameters
                lora_params = [
                    p for n, p in self._model.named_parameters() if "lora_" in n and p.requires_grad
                ]
                if not lora_params:
                    return TrainingResult(
                        adapter_name=adapter_name,
                        status="failed",
                        error="No LoRA parameters found",
                    )

                optimizer = torch.optim.AdamW(
                    lora_params,
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                )
                device = next(self._model.parameters()).device
                self._model.train()

                step = 0
                last_loss = 0.0
                for _epoch in range(config.epochs):
                    for batch in loader:
                        if step >= config.max_steps:
                            break

                        input_ids = batch["input_ids"].to(device)
                        labels = batch["labels"].to(device)
                        output = self._model(input_ids)
                        logits = output["logits"] if isinstance(output, dict) else output

                        loss = torch.nn.functional.cross_entropy(
                            logits[:, :-1].reshape(-1, logits.size(-1)),
                            labels[:, 1:].reshape(-1),
                            ignore_index=-100,
                        )

                        loss.backward()  # type: ignore[no-untyped-call]
                        torch.nn.utils.clip_grad_norm_(lora_params, config.gradient_clip)
                        optimizer.step()
                        optimizer.zero_grad()

                        last_loss = float(loss.item())
                        step += 1

                        if step % 5 == 0:
                            logger.info(
                                "[TrainingRuntime] %s step %d/%d loss=%.4f",
                                adapter_name,
                                step,
                                config.max_steps,
                                last_loss,
                            )

                # Save checkpoint
                adapter_state = LoRAManager.get_lora_state_dict(self._model)
                from pathlib import Path

                save_dir = Path(config.checkpoint_dir) / adapter_name
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / "lora_adapter.pt"
                torch.save(adapter_state, save_path)

                # Reset active adapter
                LoRAManager.set_active_adapter(self._model, None)

                training_ms = (time.monotonic() - start) * 1000
                return TrainingResult(
                    adapter_name=adapter_name,
                    checkpoint_path=str(save_path),
                    steps_completed=step,
                    final_loss=last_loss,
                    training_time_ms=training_ms,
                    adapter_state=adapter_state,
                    status="completed",
                    metrics={"epochs": config.epochs, "rank": config.rank},
                )

            except Exception as e:
                logger.error("[TrainingRuntime] SFT failed for '%s': %s", adapter_name, e)
                return TrainingResult(
                    adapter_name=adapter_name,
                    status="failed",
                    error=str(e),
                    training_time_ms=(time.monotonic() - start) * 1000,
                )

        return await asyncio.to_thread(_do_train)

    async def _train_dpo(
        self,
        adapter_name: str,
        training_pairs: list[tuple[str, str, str]],
        config: TrainingConfig,
    ) -> TrainingResult:
        """Execute DPO training.

        Wraps the existing hbllm.training.dpo pipeline.
        training_pairs: list of (prompt, chosen_response, rejected_response)
        """
        import asyncio

        start = time.monotonic()

        def _do_train() -> TrainingResult:
            try:
                import torch

                from hbllm.modules.lora import LoRAManager
                from hbllm.training.dpo import compute_dpo_loss, get_batch_logps

                if not training_pairs:
                    return TrainingResult(
                        adapter_name=adapter_name,
                        status="failed",
                        error="No training pairs provided",
                    )

                # Inject LoRA
                LoRAManager.inject(self._model, r=config.rank)
                LoRAManager.add_adapter(self._model, adapter_name)
                LoRAManager.set_active_adapter(self._model, adapter_name)

                lora_params = [
                    p for n, p in self._model.named_parameters() if "lora_" in n and p.requires_grad
                ]
                if not lora_params:
                    return TrainingResult(
                        adapter_name=adapter_name,
                        status="failed",
                        error="No LoRA parameters found",
                    )

                optimizer = torch.optim.AdamW(
                    lora_params,
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                )
                device = next(self._model.parameters()).device
                self._model.train()

                step = 0
                last_loss = 0.0

                for prompt, chosen, rejected in training_pairs:
                    if step >= config.max_steps:
                        break

                    try:
                        # Tokenize
                        c_ids = self._tokenizer.encode(f"{prompt}\n{chosen}")[:512]
                        r_ids = self._tokenizer.encode(f"{prompt}\n{rejected}")[:512]
                        chosen_ids = torch.tensor([c_ids], dtype=torch.long, device=device)
                        rejected_ids = torch.tensor([r_ids], dtype=torch.long, device=device)

                        # Reference log-probs (no LoRA)
                        LoRAManager.set_active_adapter(self._model, None)
                        with torch.no_grad():
                            ref_c_out = self._model(chosen_ids)
                            ref_r_out = self._model(rejected_ids)
                            ref_c_logits = (
                                ref_c_out["logits"] if isinstance(ref_c_out, dict) else ref_c_out
                            )
                            ref_r_logits = (
                                ref_r_out["logits"] if isinstance(ref_r_out, dict) else ref_r_out
                            )
                            ref_c_logps = get_batch_logps(ref_c_logits, chosen_ids)
                            ref_r_logps = get_batch_logps(ref_r_logits, rejected_ids)

                        # Policy log-probs (with LoRA)
                        LoRAManager.set_active_adapter(self._model, adapter_name)
                        c_out = self._model(chosen_ids)
                        r_out = self._model(rejected_ids)
                        c_logits = c_out["logits"] if isinstance(c_out, dict) else c_out
                        r_logits = r_out["logits"] if isinstance(r_out, dict) else r_out
                        pol_c_logps = get_batch_logps(c_logits, chosen_ids)
                        pol_r_logps = get_batch_logps(r_logits, rejected_ids)

                        losses, _, _ = compute_dpo_loss(
                            pol_c_logps, pol_r_logps, ref_c_logps, ref_r_logps
                        )
                        loss = losses.mean()
                        loss.backward()  # type: ignore[no-untyped-call]

                        torch.nn.utils.clip_grad_norm_(lora_params, config.gradient_clip)
                        optimizer.step()
                        optimizer.zero_grad()

                        last_loss = float(loss.item())
                        step += 1

                    except Exception as pair_err:
                        logger.warning("[TrainingRuntime] DPO pair failed: %s", pair_err)
                        continue

                # Save
                adapter_state = LoRAManager.get_lora_state_dict(
                    self._model, adapter_name=adapter_name
                )
                LoRAManager.set_active_adapter(self._model, None)

                training_ms = (time.monotonic() - start) * 1000
                return TrainingResult(
                    adapter_name=adapter_name,
                    steps_completed=step,
                    final_loss=last_loss,
                    training_time_ms=training_ms,
                    adapter_state=adapter_state,
                    status="completed",
                    metrics={"pairs": len(training_pairs), "rank": config.rank},
                )

            except Exception as e:
                logger.error("[TrainingRuntime] DPO failed for '%s': %s", adapter_name, e)
                return TrainingResult(
                    adapter_name=adapter_name,
                    status="failed",
                    error=str(e),
                    training_time_ms=(time.monotonic() - start) * 1000,
                )

        return await asyncio.to_thread(_do_train)

    # ── Job management ────────────────────────────────────────────────────

    @property
    def active_job_count(self) -> int:
        return len(self._active_jobs)

    @property
    def completed_job_count(self) -> int:
        return len(self._completed_jobs)

    def get_job(self, job_id: str) -> TrainingJob | None:
        return self._active_jobs.get(job_id)

    def list_completed(self) -> list[TrainingJob]:
        return list(self._completed_jobs)
