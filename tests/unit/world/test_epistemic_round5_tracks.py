from __future__ import annotations

"""Unit tests for Round 5 Epistemic & Cortical Neuro-Symbolic Enhancements."""

from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.cortex_deliberation import (
    BasalGangliaGatingCircuit,
    BasalGangliaPathway,
)


class TestTrack1BasalGangliaTriPathwayArbitration:
    """Track 1: Basal Ganglia D1/D2/STN arbitration."""

    def test_direct_go_pathway_selection(self):
        bg = BasalGangliaGatingCircuit()
        utilities = {1: 0.9, 2: 0.2, 3: 0.1}

        winning, all_dec = bg.arbitrate_candidates(
            candidate_utilities=utilities,
            dopamine_level=1.2,
            acc_conflict=0.1,
            threat_looming=0.0,
        )

        assert winning is not None
        assert winning.action == 1
        assert winning.pathway == BasalGangliaPathway.DIRECT_GO
        assert not winning.is_suppressed
        assert winning.d1_go_drive > winning.d2_nogo_drive
        assert winning.net_drive > 0.0

    def test_indirect_nogo_pathway_suppression(self):
        bg = BasalGangliaGatingCircuit()
        utilities = {1: 0.8, 2: 0.6}
        # Action 1 was lethal / punished in episodic memory
        hab_penalties = {1: 1.0, 2: 0.0}

        winning, all_dec = bg.arbitrate_candidates(
            candidate_utilities=utilities,
            dopamine_level=0.5,  # Low dopamine elevates D2 sensitivity
            acc_conflict=0.1,
            threat_looming=0.0,
            habenular_inhibitions=hab_penalties,
        )

        dec_1 = next(d for d in all_dec if d.action == 1)
        assert dec_1.pathway == BasalGangliaPathway.INDIRECT_NOGO
        assert dec_1.is_suppressed
        assert dec_1.d2_nogo_drive > dec_1.d1_go_drive

        # Action 2 should win as it was not punished
        assert winning is not None
        assert winning.action == 2

    def test_hyperdirect_stn_emergency_brake(self):
        bg = BasalGangliaGatingCircuit(brake_threshold=0.65)
        utilities = {1: 0.9, 2: 0.8}

        # Acute conflict from ACC and looming collision threat
        winning, all_dec = bg.arbitrate_candidates(
            candidate_utilities=utilities,
            dopamine_level=1.0,
            acc_conflict=0.9,
            threat_looming=0.85,
        )

        # STN brake should trigger global halt
        assert all(d.pathway == BasalGangliaPathway.HYPERDIRECT_BRAKE for d in all_dec)
        assert all(d.is_suppressed for d in all_dec)
        assert winning is not None
        assert winning.pathway == BasalGangliaPathway.HYPERDIRECT_BRAKE

    def test_engine_basal_ganglia_integration(self):
        engine = AutonomousEpistemicEngine()

        utilities = {1: 0.7, 2: 0.3}

        winning, all_dec = engine.arbitrate_basal_ganglia_action(
            candidate_utilities=utilities,
            acc_conflict=0.2,
            threat_looming=0.0,
        )

        assert winning is not None
        assert winning.action == 1
        assert winning.pathway == BasalGangliaPathway.DIRECT_GO
        assert winning.to_causal_predicate().variable == "basal_ganglia:pathway"
        assert len(winning.to_transaction_operations()) > 0
