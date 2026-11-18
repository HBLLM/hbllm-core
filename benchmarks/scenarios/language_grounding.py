from __future__ import annotations

"""Language Grounding Benchmark — text descriptions → entity matching.

Tests that text observations produce features that can be fused with
visual features, and that concept formation spans modalities.
"""

import time

import numpy as np

from benchmarks.sge_integration_benchmark import BenchmarkMetrics, _make_engine


class LanguageGroundingScenario:
    """Multi-modal scenario: visual entities + text descriptions."""

    def run(self) -> BenchmarkMetrics:
        from hbllm.hcir.grounding.features.composite_observation_adapter import (
            CompositeObservationAdapter,
        )
        from hbllm.hcir.grounding.features.feature_extractor import PerceptualFeatureVector
        from hbllm.hcir.grounding.features.feature_fusion import (
            FeatureFusion,
            group_features_by_entity,
        )
        from hbllm.hcir.world.autonomous_epistemic_engine import EmbodiedObservation

        metrics = BenchmarkMetrics(scenario_name="language_grounding")
        engine, bridge = _make_engine()
        adapter = CompositeObservationAdapter()
        fuser = FeatureFusion()
        t0 = time.perf_counter()

        try:
            # Phase 1: Visual observation
            grid = np.zeros((6, 6), dtype=int)
            grid[2, 3] = 7
            obs = EmbodiedObservation(
                lattice=grid,
                available_actions=[1, 2, 3, 4],
                entities=[{"position": (2.0, 3.0), "color": 7, "role": "widget"}],
            )
            engine.step(obs)
            metrics.steps += 1

            # Phase 2: Text observation describing the same entity
            text_envelope, text_features = adapter.adapt_text(
                "A red widget at position (2, 3) on the workbench.",
                named_entities={"OBJECT": ["red widget"], "LOCATION": ["workbench"]},
                embeddings=[2.0, 3.0, 7.0, 1.0],
            )
            metrics.features_extracted += len(text_features)

            # Phase 3: Fuse visual + text features
            all_features: list[PerceptualFeatureVector] = []
            all_features.extend(bridge.state.last_feature_vectors)
            all_features.extend(text_features)

            if all_features:
                bundles = group_features_by_entity(all_features)
                fused = fuser.fuse_bundles(bundles)
                metrics.concepts_formed = len(fused)

                # Verify cross-modal fusion occurred
                multimodal_bundles = [b for b in bundles if b.is_multimodal]
                if multimodal_bundles:
                    metrics.features_extracted += sum(b.total_features for b in multimodal_bundles)

            # Phase 4: More visual steps
            for step in range(4):
                grid = np.zeros((6, 6), dtype=int)
                grid[2 + step % 2, 3] = 7
                obs = EmbodiedObservation(
                    lattice=grid,
                    available_actions=[1, 2, 3, 4],
                    entities=[
                        {"position": (float(2 + step % 2), 3.0), "color": 7, "role": "widget"},
                    ],
                )
                engine.step(obs)
                metrics.steps += 1

            metrics.entities_tracked = len(bridge.tracker.get_active_candidates())

        except Exception as exc:
            metrics.errors.append(f"Scenario failed: {exc}")

        elapsed = (time.perf_counter() - t0) * 1000
        metrics.total_time_ms = elapsed
        metrics.mean_step_time_ms = elapsed / max(1, metrics.steps)
        return metrics
