from __future__ import annotations

from enum import Enum


class ModalitySource(str, Enum):
    """Sensory or computational source modality for observations and evidence."""

    VISION = "vision"
    AUDIO = "audio"
    TEXT = "text"
    KINESTHETIC = "kinesthetic"
    PROPRIOCEPTIVE = "proprioceptive"
    LIDAR = "lidar"
    ENVIRONMENTAL_SENSOR = "environmental_sensor"
    SEMANTIC_LLM_ADAPTER = "semantic_llm_adapter"
    SYSTEM_EVENT = "system_event"


class ClaimOrigin(str, Enum):
    """The origin / provenance of a grounding claim."""

    DIRECT_SENSORY = "direct_sensory"
    REPORTED_BY_AGENT = "reported_by_agent"
    INFERRED_BY_REASONING = "inferred_by_reasoning"
    HYPOTHETICAL = "hypothetical"


class DialogueAct(str, Enum):
    """Communicative intention category of a natural language utterance."""

    INFORM = "inform"
    REQUEST = "request"
    COMMAND = "command"
    QUESTION = "question"
    CORRECTION = "correction"
    NEGATION = "negation"


class Polarity(str, Enum):
    """Strictly binary logical truth intent of a claim.

    Epistemic uncertainty and ambiguity are modeled separately as continuous
    metrics on BeliefState, never as a ternary logical polarity.
    """

    AFFIRMATIVE = "affirmative"
    NEGATIVE = "negative"


class TrafficClass(str, Enum):
    """Quality of Service and buffering policies for sensor / observation channels."""

    CLASS_HIGH_FREQ = "class_high_freq"  # Video / LiDAR: bounded ring, latest-wins decimation
    CLASS_STREAMING = "class_streaming"  # Audio: bounded latency queue, flow-control signals
    CLASS_TELEMETRY = "class_telemetry"  # IMU / proprioception: monotonic series with gap detection
    CLASS_DISCRETE = "class_discrete"  # User text, discrete UI events: durable queue, backpressure
    CLASS_TRANSACTION = (
        "class_transaction"  # Transaction proposals: zero-drop, blocking backpressure
    )


class OverflowPolicy(str, Enum):
    """Behavior when a channel or buffer reaches capacity."""

    DROP_OLDEST = "drop_oldest"  # Overwrite oldest unconsumed item (classic ring)
    DROP_NEWEST = "drop_newest"  # Discard newly arriving packet
    BLOCK_PRODUCER = "block_producer"  # Apply synchronous/asynchronous backpressure
    REJECT_WITH_ERROR = "reject_with_error"  # Raise BufferOverflowError


class EvidenceRelation(str, Enum):
    """The relation between a piece of evidence and a proposition / claim."""

    SUPPORTS = "supports"
    REFUTES = "refutes"
    NEUTRAL = "neutral"


class ConceptLifecycle(str, Enum):
    """Lifecycle of grounded concepts across working store and CognitiveGraph."""

    EMERGING = "emerging"
    PROVISIONAL = "provisional"
    GROUNDED = "grounded"  # Sufficient empirical support in SGE Working Concept Store
    CONSOLIDATED = "consolidated"  # Sufficient stability & utility for promotion consideration
    CANONICAL = "canonical"  # Formally committed via TransactionManager into CognitiveGraph
    REVISED = "revised"
    MERGED_SPLIT_DEPRECATED = "merged_split_deprecated"


class CausalStatus(str, Enum):
    """Pearl's causal ladder status for relational hypotheses and schemas."""

    OBSERVED_ASSOCIATION = "observed_association"  # P(B | A) correlational
    PREDICTIVE_NON_CAUSAL = "predictive_non_causal"  # Forward predictive, unverified intervention
    CAUSAL_HYPOTHESIS = "causal_hypothesis"  # Structural hypothesis formulated
    CAUSAL_EFFECT_IDENTIFIED = (
        "causal_effect_identified"  # Identified under observational assumptions (e.g. backdoor)
    )
    INTERVENTIONALLY_SUPPORTED = (
        "interventionally_supported"  # P(B | do(A)) experimentally measured via trials
    )
    COUNTERFACTUAL_MODEL_AVAILABLE = (
        "counterfactual_model_available"  # SCM parameterized to support counterfactual queries
    )
    CAUSALLY_IDENTIFIED = "causally_identified"  # Compatibility alias for Rung 3 SCM


class DecisionOutcome(str, Enum):
    """Epistemic disposition outcome for a GroundingClaim."""

    ACCEPT = "accept"
    REJECT = "reject"
    DEFER = "defer"
    REQUEST_EVIDENCE = "request_evidence"


class ActionLifecycleState(str, Enum):
    """State transition lifecycle for grounded action execution."""

    PROPOSED = "proposed"
    VALIDATED = "validated"
    AUTHORIZED = "authorized"
    DISPATCHED = "dispatched"
    EXECUTION_REPORTED = "execution_reported"
    OUTCOME_VERIFIED = "outcome_verified"
    COMPLETED = "completed"
    FAILED = "failed"
    UNCERTAIN = "uncertain"
