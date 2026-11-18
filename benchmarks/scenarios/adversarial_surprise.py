from __future__ import annotations

"""Adversarial Surprise Benchmark — deliberately violates predictions.

Tests the belief revision pipeline: prediction → violation → surprise →
schema degradation → epistemic curiosity goal.
"""

import time

import numpy as np

from benchmarks.sge_integration_benchmark import BenchmarkMetrics, _make_engine


class AdversarialSurpriseScenario:
    """Sets up predictions and deliberately violates them to test belief revision."""

    def run(self) -> BenchmarkMetrics:
        from hbllm.hcir.grounding.contracts.enums import CausalStatus
        from hbllm.hcir.world.autonomous_epistemic_engine import EmbodiedObservation

        metrics = BenchmarkMetrics(scenario_name="adversarial_surprise")
        engine, bridge = _make_engine()
        t0 = time.perf_counter()

        try:
            # Phase 1: Establish baseline entity
            grid = np.zeros((8, 8), dtype=int)
            grid[4, 4] = 8
            obs = EmbodiedObservation(
                lattice=grid,
                available_actions=[1, 2, 3, 4],
                entities=[{"position": (4.0, 4.0), "color": 8, "role": "target"}],
            )
            engine.step(obs)
            metrics.steps += 1

            candidates = bridge.tracker.get_active_candidates()
            if not candidates:
                metrics.errors.append("No candidates tracked after initial observation")
                return metrics
            target_ref = candidates[0].candidate_ref

            # Phase 2: Set up fragile causal hypotheses
            schema_push = bridge.schema_inducer.record_observation(
                action_type="PUSH",
                target_role="target",
                precondition={},
                outcome_event=None,
            )
            schema_push.causal_status = CausalStatus.INTERVENTIONALLY_SUPPORTED
            schema_push.p_effect_given_do_action = 0.90
            schema_push.expected_effect = {"post_state": {"speed": 1.0, "moved": True}}
            bridge.predictor.register_schema(schema_push)

            # Phase 3: Set up prediction and dispatch action
            bridge.state.active_predictions = {
                target_ref.id: bridge.predictor.predict_next_state(
                    {target_ref.id: {"ref": target_ref, "position": (4.0, 4.0)}},
                    action_type="PUSH",
                    target_entity_ref=target_ref,
                )[target_ref.id]
            }

            world_st = {
                "tracked_entities": {
                    target_ref.id: {"confidence": 0.95, "position": (4.0, 4.0)},
                },
            }
            act, rej = bridge.process_motor(
                ctx=None,  # type: ignore[arg-type]
                proposed_action="PUSH",
                action_data={"target_id": target_ref.id, "speed": 1.0},
                world_state=world_st,
            )
            if act:
                metrics.actions_authorised += 1
            if rej:
                metrics.actions_rejected += 1

            # Phase 4: ADVERSARIAL VIOLATION — entity stays put
            actual_obs = {target_ref.id: {"position": (4.0, 4.0), "speed": 0.0}}
            assim_res = bridge.process_assimilation(
                ctx=None,  # type: ignore[arg-type]
                current_entity_states=actual_obs,
            )
            metrics.surprise_alerts = assim_res.get("sge_surprise_alerts", 0)

            # Validate surprise detection
            if metrics.surprise_alerts < 1:
                metrics.errors.append("Expected surprise alert but got none")
            max_surprise = assim_res.get("sge_max_surprise", 0.0)
            if max_surprise < 0.3:
                metrics.errors.append(f"Expected surprise > 0.3 but got {max_surprise:.3f}")

            # Validate schema degradation
            if schema_push.p_effect_given_do_action >= 0.90:
                metrics.errors.append(
                    f"Schema not degraded: p={schema_push.p_effect_given_do_action:.3f}"
                )
            if schema_push.confounder_risk <= 0.5:
                metrics.errors.append(
                    f"Confounder risk not raised: {schema_push.confounder_risk:.3f}"
                )

            # Phase 5: Continue observation to verify recovery
            for step in range(3):
                grid = np.zeros((8, 8), dtype=int)
                grid[4, 4] = 8  # Entity still fixed
                obs = EmbodiedObservation(
                    lattice=grid,
                    available_actions=[1, 2, 3, 4],
                    entities=[{"position": (4.0, 4.0), "color": 8, "role": "target"}],
                )
                engine.step(obs)
                metrics.steps += 1

            metrics.entities_tracked = len(bridge.tracker.get_active_candidates())
            metrics.concepts_formed = len(bridge.concept_store.get_all_concepts())

        except Exception as exc:
            metrics.errors.append(f"Scenario failed: {exc}")

        elapsed = (time.perf_counter() - t0) * 1000
        metrics.total_time_ms = elapsed
        metrics.mean_step_time_ms = elapsed / max(1, metrics.steps)
        return metrics
