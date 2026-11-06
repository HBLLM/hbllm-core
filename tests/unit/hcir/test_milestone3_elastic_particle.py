"""Unit Test Suite for Milestone 3 Phase 3: Elastic Particle Dynamics & Angle-of-Incidence Deflection."""

import ast
from pathlib import Path

import numpy as np

from experiments.benchmarks.task_generators.elastic_particle_generator import (
    ElasticParticleTaskGenerator,
)
from hbllm.hcir.world.elastic_particle_induction import (
    ElasticParticleDeflectionOperator,
    InductiveParticleDeflectionSynthesizer,
    ParticleKinematicsSimulator,
)
from hbllm.hcir.world.representation_expansion import RepresentationExpansionEngine


def test_generator_ast_independence() -> None:
    """Verify that ElasticParticleTaskGenerator has 0 imports from hbllm/hcir solver internals."""
    gen_file = (
        Path(__file__).resolve().parents[3]
        / "experiments"
        / "benchmarks"
        / "task_generators"
        / "elastic_particle_generator.py"
    )
    assert gen_file.exists(), f"File {gen_file} not found"

    tree = ast.parse(gen_file.read_text(encoding="utf-8"))
    forbidden_prefixes = ("hbllm", "core.hbllm")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for prefix in forbidden_prefixes:
                    assert not alias.name.startswith(prefix), (
                        f"Forbidden import '{alias.name}' found in task generator"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for prefix in forbidden_prefixes:
                    assert not node.module.startswith(prefix), (
                        f"Forbidden import from '{node.module}' found in task generator"
                    )


def test_particle_kinematics_simulator_deflection() -> None:
    """Verify 90-degree elastic deflection upon striking barrier."""
    grid = np.zeros((8, 8), dtype=int)
    grid[0, 1] = 2  # Emitter
    grid[4, 5] = 5  # Barrier

    trajectory = ParticleKinematicsSimulator.simulate_trajectory(
        grid=grid,
        start_coord=(0, 1),
        initial_velocity=(1, 1),
        barrier_colors={5},
        deflection_mode="horizontal_flip",
    )
    # Trajectory should advance diagonally to (3, 4), reflect at (4, 5), then advance (4, 3), (5, 2), (6, 1)
    assert (1, 2) in trajectory
    assert (3, 4) in trajectory
    assert (4, 3) in trajectory
    assert (5, 2) in trajectory
    assert (6, 1) in trajectory


def test_particle_kinematics_simulator_absorption() -> None:
    """Verify ray terminates upon striking absorbing target."""
    grid = np.zeros((8, 8), dtype=int)
    grid[0, 0] = 2  # Emitter
    grid[3, 3] = 7  # Absorbing target

    trajectory = ParticleKinematicsSimulator.simulate_trajectory(
        grid=grid,
        start_coord=(0, 0),
        initial_velocity=(1, 1),
        barrier_colors=set(),
        absorbing_colors={7},
    )
    # Trajectory advances (1, 1), (2, 2) and halts before (3, 3)
    assert (1, 1) in trajectory
    assert (2, 2) in trajectory
    assert (3, 3) not in trajectory
    assert (4, 4) not in trajectory


def test_inductive_synthesizer_from_residuals() -> None:
    """Verify inductive discovery of particle kinematics parameters from demonstrations."""
    # Demo 0
    x0 = np.zeros((8, 8), dtype=int)
    x0[0, 1] = 2  # Emitter
    x0[4, 5] = 5  # Barrier
    op_truth = ElasticParticleDeflectionOperator(
        name="ground_truth",
        emitter_color=2,
        beam_color=6,
        barrier_colors=(5,),
        initial_velocity=(1, 1),
        deflection_mode="horizontal_flip",
    )
    b0 = op_truth.propose({}, {})[0]
    y0 = op_truth.apply(x0, b0)

    # Demo 1 (different barrier position)
    x1 = np.zeros((8, 8), dtype=int)
    x1[0, 0] = 2
    x1[3, 3] = 5
    y1 = op_truth.apply(x1, b0)

    op_induced = InductiveParticleDeflectionSynthesizer.synthesize_from_residuals(
        [(x0, y0), (x1, y1)]
    )
    assert op_induced is not None
    assert op_induced.emitter_color == 2
    assert op_induced.beam_color == 6
    assert 5 in op_induced.barrier_colors
    assert op_induced.initial_velocity == (1, 1)


def test_end_to_end_elastic_particle_benchmark_suite() -> None:
    """Verify suite generation and end-to-end synthesis across families."""
    suite = ElasticParticleTaskGenerator.generate_benchmark_suite(seed=123)
    assert len(suite) == 20

    # Test single task from single_deflection_billiard
    task = next(t for t in suite if t.family == "single_deflection_billiard")
    trace = RepresentationExpansionEngine.synthesize_and_predict(
        list(task.train_pairs), task.test_input, task_id=task.task_id, family=task.family
    )
    assert trace.phase2_solved is True
    assert trace.phase2_predicted_test is not None
    assert np.array_equal(trace.phase2_predicted_test, task.test_output)
