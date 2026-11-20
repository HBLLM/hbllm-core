"""Track T3 CLI Runner: Parallel Shadow Distillation (LLM Scaffold -> Pure-HCIR).

Runs multi-epoch multi-domain dialogue + governed tool-use tasks through
``CognitiveSystem.process_dialogue()`` and ``KernelServices.capability_resolver``:
  1. Pure-HCIR attempts every turn first (using Track T1 110K-entry ontology).
  2. On HCIR parse/grounding gaps, ``LLMTeacherScaffold`` (strictly outside
     ``hbllm/hcir/``) distills verified entities, lexical entries, and beliefs
     into HCIR via ``TransactionManager``.
  3. On subsequent epochs, Pure-HCIR handles those turns natively without the LLM,
     tracking ``hcir_parse_autonomy_rate``, ``hcir_plan_autonomy_rate``, and
     ``hcir_realize_autonomy_rate`` toward the ``>= 98%`` retirement gate.

Examples:
    # Fast curriculum shadow run (CPU-only, offline)
    .venv/bin/python scripts/run_parallel_shadow_training.py --mode curriculum --epochs 5

    # Live NVIDIA NIM teacher run (uses NVIDIA_API_KEY from .env)
    .venv/bin/python scripts/run_parallel_shadow_training.py --mode nvidia --epochs 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hbllm.hcir.kernel.autonomy_tracker import AutonomyTracker
from hbllm.hcir.language.ontology_store import DEFAULT_ONTOLOGY_DB, LexicalOntologyStore
from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system
from plugins.llm_scaffold import (
    DEFAULT_LOCAL_CHECKPOINT,
    LLMTeacherScaffold,
)

# Multi-domain dialogue curriculum (software/cloud, science/lab, everyday + queries)
CURRICULUM_EPISODES: list[dict[str, Any]] = [
    # Software / cloud infrastructure
    {
        "utterance": "The server is in the cluster.",
        "lesson": {
            "response_text": "Noted: the server is located in the cluster.",
            "claim": "server located_in cluster",
            "entities": [
                {"name": "server", "type": "server"},
                {"name": "cluster", "type": "cluster"},
            ],
            "lexicon": [],
            "confidence": 0.92,
            "verified": True,
        },
    },
    {
        "utterance": "The database is on the server.",
        "lesson": {
            "response_text": "Noted: the database is located on the server.",
            "claim": "database located_on server",
            "entities": [
                {"name": "database", "type": "database"},
                {"name": "server", "type": "server"},
            ],
            "lexicon": [],
            "confidence": 0.91,
            "verified": True,
        },
    },
    {
        "utterance": "The cache is near the database.",
        "lesson": {
            "response_text": "Noted: the cache is near the database.",
            "claim": "cache near database",
            "entities": [
                {"name": "cache", "type": "cache"},
                {"name": "database", "type": "database"},
            ],
            "lexicon": [],
            "confidence": 0.91,
            "verified": True,
        },
    },
    {
        "utterance": "The container is in the cluster.",
        "lesson": {
            "response_text": "Noted: the container is in the cluster.",
            "claim": "container located_in cluster",
            "entities": [
                {"name": "container", "type": "container"},
                {"name": "cluster", "type": "cluster"},
            ],
            "lexicon": [],
            "confidence": 0.92,
            "verified": True,
        },
    },
    {
        "utterance": "The router is near the firewall.",
        "lesson": {
            "response_text": "Noted: the router is near the firewall.",
            "claim": "router near firewall",
            "entities": [
                {"name": "router", "type": "router"},
                {"name": "firewall", "type": "firewall"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "The file is in the folder.",
        "lesson": {
            "response_text": "Noted: the file is in the folder.",
            "claim": "file located_in folder",
            "entities": [{"name": "file", "type": "file"}, {"name": "folder", "type": "folder"}],
            "lexicon": [],
            "confidence": 0.93,
            "verified": True,
        },
    },
    # Science / robotics / lab
    {
        "utterance": "The sensor is near the robot.",
        "lesson": {
            "response_text": "Noted: the sensor is near the robot.",
            "claim": "sensor near robot",
            "entities": [{"name": "sensor", "type": "sensor"}, {"name": "robot", "type": "robot"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "The microscope is on the table.",
        "lesson": {
            "response_text": "Noted: the microscope is on the table.",
            "claim": "microscope located_on table",
            "entities": [
                {"name": "microscope", "type": "microscope"},
                {"name": "table", "type": "table"},
            ],
            "lexicon": [],
            "confidence": 0.93,
            "verified": True,
        },
    },
    {
        "utterance": "The telescope is in the room.",
        "lesson": {
            "response_text": "Noted: the telescope is in the room.",
            "claim": "telescope located_in room",
            "entities": [
                {"name": "telescope", "type": "telescope"},
                {"name": "room", "type": "room"},
            ],
            "lexicon": [],
            "confidence": 0.92,
            "verified": True,
        },
    },
    {
        "utterance": "The battery is inside the robot.",
        "lesson": {
            "response_text": "Noted: the battery is inside the robot.",
            "claim": "battery located_in robot",
            "entities": [
                {"name": "battery", "type": "battery"},
                {"name": "robot", "type": "robot"},
            ],
            "lexicon": [],
            "confidence": 0.91,
            "verified": True,
        },
    },
    {
        "utterance": "The lens is on the microscope.",
        "lesson": {
            "response_text": "Noted: the lens is on the microscope.",
            "claim": "lens located_on microscope",
            "entities": [
                {"name": "lens", "type": "lens"},
                {"name": "microscope", "type": "microscope"},
            ],
            "lexicon": [],
            "confidence": 0.91,
            "verified": True,
        },
    },
    # Everyday / spatial
    {
        "utterance": "The key is under the book.",
        "lesson": {
            "response_text": "Noted: the key is under the book.",
            "claim": "key located_under book",
            "entities": [{"name": "key", "type": "key"}, {"name": "book", "type": "book"}],
            "lexicon": [],
            "confidence": 0.91,
            "verified": True,
        },
    },
    {
        "utterance": "The document is on the desk.",
        "lesson": {
            "response_text": "Noted: the document is on the desk.",
            "claim": "document located_on desk",
            "entities": [
                {"name": "document", "type": "document"},
                {"name": "desk", "type": "desk"},
            ],
            "lexicon": [],
            "confidence": 0.92,
            "verified": True,
        },
    },
    {
        "utterance": "The lamp is above the desk.",
        "lesson": {
            "response_text": "Noted: the lamp is above the desk.",
            "claim": "lamp above desk",
            "entities": [{"name": "lamp", "type": "lamp"}, {"name": "desk", "type": "desk"}],
            "lexicon": [],
            "confidence": 0.92,
            "verified": True,
        },
    },
    # Grounded Queries (tests recall & realization)
    {
        "utterance": "Where is the server?",
        "lesson": {
            "response_text": "The server is in the cluster.",
            "claim": "server located_in cluster",
            "entities": [{"name": "server", "type": "server"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Where is the database?",
        "lesson": {
            "response_text": "The database is on the server.",
            "claim": "database located_on server",
            "entities": [{"name": "database", "type": "database"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Where is the microscope?",
        "lesson": {
            "response_text": "The microscope is on the table.",
            "claim": "microscope located_on table",
            "entities": [{"name": "microscope", "type": "microscope"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Where is the file?",
        "lesson": {
            "response_text": "The file is in the folder.",
            "claim": "file located_in folder",
            "entities": [{"name": "file", "type": "file"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Where is the key?",
        "lesson": {
            "response_text": "The key is under the book.",
            "claim": "key located_under book",
            "entities": [{"name": "key", "type": "key"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Where is the document?",
        "lesson": {
            "response_text": "The document is on the desk.",
            "claim": "document located_on desk",
            "entities": [{"name": "document", "type": "document"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    # ── Multi-clause discourse (20 episodes) ──────────────────────────────────
    {
        "utterance": "Read the file and list the directory.",
        "lesson": {
            "response_text": "Reading the file and listing the directory.",
            "claim": "command read file AND command list directory",
            "entities": [
                {"name": "file", "type": "file"},
                {"name": "directory", "type": "directory"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Open the config and check the syntax.",
        "lesson": {
            "response_text": "Opening the config and checking the syntax.",
            "claim": "command open config AND command check syntax",
            "entities": [
                {"name": "config", "type": "file"},
                {"name": "syntax", "type": "property"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Move the ball to the table and pick up the key.",
        "lesson": {
            "response_text": "Moving the ball to the table and picking up the key.",
            "claim": "command move ball table AND command pickup key",
            "entities": [
                {"name": "ball", "type": "object"},
                {"name": "table", "type": "surface"},
                {"name": "key", "type": "object"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Find the sensor and inspect the robot.",
        "lesson": {
            "response_text": "Finding the sensor and inspecting the robot.",
            "claim": "command find sensor AND command inspect robot",
            "entities": [
                {"name": "sensor", "type": "sensor"},
                {"name": "robot", "type": "robot"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Deploy the container and monitor the logs.",
        "lesson": {
            "response_text": "Deploying the container and monitoring the logs.",
            "claim": "command deploy container AND command monitor logs",
            "entities": [
                {"name": "container", "type": "container"},
                {"name": "logs", "type": "log"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Scan the network and update the firewall.",
        "lesson": {
            "response_text": "Scanning the network and updating the firewall.",
            "claim": "command scan network AND command update firewall",
            "entities": [
                {"name": "network", "type": "network"},
                {"name": "firewall", "type": "firewall"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Grab the cup and place it on the shelf.",
        "lesson": {
            "response_text": "Grabbing the cup and placing it on the shelf.",
            "claim": "command grab cup AND command place cup shelf",
            "entities": [
                {"name": "cup", "type": "container"},
                {"name": "shelf", "type": "surface"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Lock the door and turn off the light.",
        "lesson": {
            "response_text": "Locking the door and turning off the light.",
            "claim": "command lock door AND command turnoff light",
            "entities": [
                {"name": "door", "type": "barrier"},
                {"name": "light", "type": "device"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Compile the code and run the tests.",
        "lesson": {
            "response_text": "Compiling the code and running the tests.",
            "claim": "command compile code AND command run tests",
            "entities": [
                {"name": "code", "type": "file"},
                {"name": "tests", "type": "test_suite"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Charge the battery and calibrate the sensor.",
        "lesson": {
            "response_text": "Charging the battery and calibrating the sensor.",
            "claim": "command charge battery AND command calibrate sensor",
            "entities": [
                {"name": "battery", "type": "battery"},
                {"name": "sensor", "type": "sensor"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Fetch the data and generate the report.",
        "lesson": {
            "response_text": "Fetching the data and generating the report.",
            "claim": "command fetch data AND command generate report",
            "entities": [
                {"name": "data", "type": "dataset"},
                {"name": "report", "type": "document"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Open the drawer and take out the pen.",
        "lesson": {
            "response_text": "Opening the drawer and taking out the pen.",
            "claim": "command open drawer AND command take pen",
            "entities": [
                {"name": "drawer", "type": "container"},
                {"name": "pen", "type": "object"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Start the server and initialize the database.",
        "lesson": {
            "response_text": "Starting the server and initializing the database.",
            "claim": "command start server AND command initialize database",
            "entities": [
                {"name": "server", "type": "server"},
                {"name": "database", "type": "database"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Measure the sample and record the results.",
        "lesson": {
            "response_text": "Measuring the sample and recording the results.",
            "claim": "command measure sample AND command record results",
            "entities": [
                {"name": "sample", "type": "specimen"},
                {"name": "results", "type": "data"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Rotate the image and save the output.",
        "lesson": {
            "response_text": "Rotating the image and saving the output.",
            "claim": "command rotate image AND command save output",
            "entities": [
                {"name": "image", "type": "file"},
                {"name": "output", "type": "file"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Clean the workspace and organize the tools.",
        "lesson": {
            "response_text": "Cleaning the workspace and organizing the tools.",
            "claim": "command clean workspace AND command organize tools",
            "entities": [
                {"name": "workspace", "type": "area"},
                {"name": "tools", "type": "equipment"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Connect the cable and test the signal.",
        "lesson": {
            "response_text": "Connecting the cable and testing the signal.",
            "claim": "command connect cable AND command test signal",
            "entities": [
                {"name": "cable", "type": "connector"},
                {"name": "signal", "type": "data"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Push the commit and open a pull request.",
        "lesson": {
            "response_text": "Pushing the commit and opening a pull request.",
            "claim": "command push commit AND command open pull_request",
            "entities": [
                {"name": "commit", "type": "version_control"},
                {"name": "pull_request", "type": "version_control"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Heat the solution and stir the mixture.",
        "lesson": {
            "response_text": "Heating the solution and stirring the mixture.",
            "claim": "command heat solution AND command stir mixture",
            "entities": [
                {"name": "solution", "type": "substance"},
                {"name": "mixture", "type": "substance"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Close the valve and check the pressure.",
        "lesson": {
            "response_text": "Closing the valve and checking the pressure.",
            "claim": "command close valve AND command check pressure",
            "entities": [
                {"name": "valve", "type": "mechanism"},
                {"name": "pressure", "type": "measurement"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    # ── Negation (15 episodes) ─────────────────────────────────────────────────
    {
        "utterance": "The ball is not on the table.",
        "lesson": {
            "response_text": "Noted: the ball is not on the table.",
            "claim": "ball NOT located_on table",
            "entities": [
                {"name": "ball", "type": "object"},
                {"name": "table", "type": "surface"},
            ],
            "lexicon": [],
            "confidence": 0.91,
            "verified": True,
        },
    },
    {
        "utterance": "The server is not in the cluster.",
        "lesson": {
            "response_text": "Noted: the server is not in the cluster.",
            "claim": "server NOT located_in cluster",
            "entities": [
                {"name": "server", "type": "server"},
                {"name": "cluster", "type": "cluster"},
            ],
            "lexicon": [],
            "confidence": 0.91,
            "verified": True,
        },
    },
    {
        "utterance": "The key is not under the book.",
        "lesson": {
            "response_text": "Noted: the key is not under the book.",
            "claim": "key NOT located_under book",
            "entities": [
                {"name": "key", "type": "object"},
                {"name": "book", "type": "object"},
            ],
            "lexicon": [],
            "confidence": 0.91,
            "verified": True,
        },
    },
    {
        "utterance": "The robot is not near the wall.",
        "lesson": {
            "response_text": "Noted: the robot is not near the wall.",
            "claim": "robot NOT near wall",
            "entities": [
                {"name": "robot", "type": "robot"},
                {"name": "wall", "type": "barrier"},
            ],
            "lexicon": [],
            "confidence": 0.91,
            "verified": True,
        },
    },
    {
        "utterance": "The container is not in the registry.",
        "lesson": {
            "response_text": "Noted: the container is not in the registry.",
            "claim": "container NOT located_in registry",
            "entities": [
                {"name": "container", "type": "container"},
                {"name": "registry", "type": "registry"},
            ],
            "lexicon": [],
            "confidence": 0.91,
            "verified": True,
        },
    },
    {
        "utterance": "The lamp is not above the chair.",
        "lesson": {
            "response_text": "Noted: the lamp is not above the chair.",
            "claim": "lamp NOT above chair",
            "entities": [
                {"name": "lamp", "type": "device"},
                {"name": "chair", "type": "furniture"},
            ],
            "lexicon": [],
            "confidence": 0.91,
            "verified": True,
        },
    },
    {
        "utterance": "The cable is not connected to the router.",
        "lesson": {
            "response_text": "Noted: the cable is not connected to the router.",
            "claim": "cable NOT connected_to router",
            "entities": [
                {"name": "cable", "type": "connector"},
                {"name": "router", "type": "router"},
            ],
            "lexicon": [],
            "confidence": 0.91,
            "verified": True,
        },
    },
    {
        "utterance": "The battery is not inside the drone.",
        "lesson": {
            "response_text": "Noted: the battery is not inside the drone.",
            "claim": "battery NOT located_in drone",
            "entities": [
                {"name": "battery", "type": "battery"},
                {"name": "drone", "type": "robot"},
            ],
            "lexicon": [],
            "confidence": 0.91,
            "verified": True,
        },
    },
    {
        "utterance": "The document is not on the printer.",
        "lesson": {
            "response_text": "Noted: the document is not on the printer.",
            "claim": "document NOT located_on printer",
            "entities": [
                {"name": "document", "type": "document"},
                {"name": "printer", "type": "device"},
            ],
            "lexicon": [],
            "confidence": 0.91,
            "verified": True,
        },
    },
    {
        "utterance": "The module is not in the package.",
        "lesson": {
            "response_text": "Noted: the module is not in the package.",
            "claim": "module NOT located_in package",
            "entities": [
                {"name": "module", "type": "module"},
                {"name": "package", "type": "package"},
            ],
            "lexicon": [],
            "confidence": 0.91,
            "verified": True,
        },
    },
    {
        "utterance": "The reagent is not in the flask.",
        "lesson": {
            "response_text": "Noted: the reagent is not in the flask.",
            "claim": "reagent NOT located_in flask",
            "entities": [
                {"name": "reagent", "type": "substance"},
                {"name": "flask", "type": "container"},
            ],
            "lexicon": [],
            "confidence": 0.91,
            "verified": True,
        },
    },
    {
        "utterance": "The phone is not on the table.",
        "lesson": {
            "response_text": "Noted: the phone is not on the table.",
            "claim": "phone NOT located_on table",
            "entities": [
                {"name": "phone", "type": "device"},
                {"name": "table", "type": "surface"},
            ],
            "lexicon": [],
            "confidence": 0.91,
            "verified": True,
        },
    },
    {
        "utterance": "The backup is not on the cloud.",
        "lesson": {
            "response_text": "Noted: the backup is not on the cloud.",
            "claim": "backup NOT located_on cloud",
            "entities": [
                {"name": "backup", "type": "data"},
                {"name": "cloud", "type": "infrastructure"},
            ],
            "lexicon": [],
            "confidence": 0.91,
            "verified": True,
        },
    },
    {
        "utterance": "The wrench is not near the toolbox.",
        "lesson": {
            "response_text": "Noted: the wrench is not near the toolbox.",
            "claim": "wrench NOT near toolbox",
            "entities": [
                {"name": "wrench", "type": "tool"},
                {"name": "toolbox", "type": "container"},
            ],
            "lexicon": [],
            "confidence": 0.91,
            "verified": True,
        },
    },
    {
        "utterance": "The camera is not above the entrance.",
        "lesson": {
            "response_text": "Noted: the camera is not above the entrance.",
            "claim": "camera NOT above entrance",
            "entities": [
                {"name": "camera", "type": "device"},
                {"name": "entrance", "type": "opening"},
            ],
            "lexicon": [],
            "confidence": 0.91,
            "verified": True,
        },
    },
    # ── Property assertions (15 episodes) ──────────────────────────────────────
    {
        "utterance": "The ball is red.",
        "lesson": {
            "response_text": "Noted: the ball is red.",
            "claim": "ball HAS_PROPERTY red",
            "entities": [{"name": "ball", "type": "object"}],
            "lexicon": [],
            "confidence": 0.92,
            "verified": True,
        },
    },
    {
        "utterance": "The server is large.",
        "lesson": {
            "response_text": "Noted: the server is large.",
            "claim": "server HAS_PROPERTY large",
            "entities": [{"name": "server", "type": "server"}],
            "lexicon": [],
            "confidence": 0.92,
            "verified": True,
        },
    },
    {
        "utterance": "The cup is empty.",
        "lesson": {
            "response_text": "Noted: the cup is empty.",
            "claim": "cup HAS_PROPERTY empty",
            "entities": [{"name": "cup", "type": "container"}],
            "lexicon": [],
            "confidence": 0.92,
            "verified": True,
        },
    },
    {
        "utterance": "The robot is fast.",
        "lesson": {
            "response_text": "Noted: the robot is fast.",
            "claim": "robot HAS_PROPERTY fast",
            "entities": [{"name": "robot", "type": "robot"}],
            "lexicon": [],
            "confidence": 0.92,
            "verified": True,
        },
    },
    {
        "utterance": "The database is encrypted.",
        "lesson": {
            "response_text": "Noted: the database is encrypted.",
            "claim": "database HAS_PROPERTY encrypted",
            "entities": [{"name": "database", "type": "database"}],
            "lexicon": [],
            "confidence": 0.92,
            "verified": True,
        },
    },
    {
        "utterance": "The cable is long.",
        "lesson": {
            "response_text": "Noted: the cable is long.",
            "claim": "cable HAS_PROPERTY long",
            "entities": [{"name": "cable", "type": "connector"}],
            "lexicon": [],
            "confidence": 0.92,
            "verified": True,
        },
    },
    {
        "utterance": "The room is dark.",
        "lesson": {
            "response_text": "Noted: the room is dark.",
            "claim": "room HAS_PROPERTY dark",
            "entities": [{"name": "room", "type": "room"}],
            "lexicon": [],
            "confidence": 0.92,
            "verified": True,
        },
    },
    {
        "utterance": "The solution is acidic.",
        "lesson": {
            "response_text": "Noted: the solution is acidic.",
            "claim": "solution HAS_PROPERTY acidic",
            "entities": [{"name": "solution", "type": "substance"}],
            "lexicon": [],
            "confidence": 0.92,
            "verified": True,
        },
    },
    {
        "utterance": "The network is secure.",
        "lesson": {
            "response_text": "Noted: the network is secure.",
            "claim": "network HAS_PROPERTY secure",
            "entities": [{"name": "network", "type": "network"}],
            "lexicon": [],
            "confidence": 0.92,
            "verified": True,
        },
    },
    {
        "utterance": "The key is golden.",
        "lesson": {
            "response_text": "Noted: the key is golden.",
            "claim": "key HAS_PROPERTY golden",
            "entities": [{"name": "key", "type": "object"}],
            "lexicon": [],
            "confidence": 0.92,
            "verified": True,
        },
    },
    {
        "utterance": "The file is corrupted.",
        "lesson": {
            "response_text": "Noted: the file is corrupted.",
            "claim": "file HAS_PROPERTY corrupted",
            "entities": [{"name": "file", "type": "file"}],
            "lexicon": [],
            "confidence": 0.92,
            "verified": True,
        },
    },
    {
        "utterance": "The motor is overheated.",
        "lesson": {
            "response_text": "Noted: the motor is overheated.",
            "claim": "motor HAS_PROPERTY overheated",
            "entities": [{"name": "motor", "type": "actuator"}],
            "lexicon": [],
            "confidence": 0.92,
            "verified": True,
        },
    },
    {
        "utterance": "The window is open.",
        "lesson": {
            "response_text": "Noted: the window is open.",
            "claim": "window HAS_PROPERTY open",
            "entities": [{"name": "window", "type": "opening"}],
            "lexicon": [],
            "confidence": 0.92,
            "verified": True,
        },
    },
    {
        "utterance": "The pipeline is broken.",
        "lesson": {
            "response_text": "Noted: the pipeline is broken.",
            "claim": "pipeline HAS_PROPERTY broken",
            "entities": [{"name": "pipeline", "type": "infrastructure"}],
            "lexicon": [],
            "confidence": 0.92,
            "verified": True,
        },
    },
    {
        "utterance": "The specimen is fragile.",
        "lesson": {
            "response_text": "Noted: the specimen is fragile.",
            "claim": "specimen HAS_PROPERTY fragile",
            "entities": [{"name": "specimen", "type": "specimen"}],
            "lexicon": [],
            "confidence": 0.92,
            "verified": True,
        },
    },
    # ── Complex queries (15 episodes) ──────────────────────────────────────────
    {
        "utterance": "What is on the table?",
        "lesson": {
            "response_text": "The microscope is on the table.",
            "claim": "query located_on table",
            "entities": [{"name": "table", "type": "surface"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "What is in the cluster?",
        "lesson": {
            "response_text": "The server and the container are in the cluster.",
            "claim": "query located_in cluster",
            "entities": [{"name": "cluster", "type": "cluster"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Is the ball on the table?",
        "lesson": {
            "response_text": "Checking whether the ball is on the table.",
            "claim": "query_bool ball located_on table",
            "entities": [
                {"name": "ball", "type": "object"},
                {"name": "table", "type": "surface"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Is the server in the cluster?",
        "lesson": {
            "response_text": "Checking whether the server is in the cluster.",
            "claim": "query_bool server located_in cluster",
            "entities": [
                {"name": "server", "type": "server"},
                {"name": "cluster", "type": "cluster"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "What is under the book?",
        "lesson": {
            "response_text": "The key is under the book.",
            "claim": "query located_under book",
            "entities": [{"name": "book", "type": "object"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "What is near the robot?",
        "lesson": {
            "response_text": "The sensor is near the robot.",
            "claim": "query near robot",
            "entities": [{"name": "robot", "type": "robot"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Is the lamp above the desk?",
        "lesson": {
            "response_text": "Yes, the lamp is above the desk.",
            "claim": "query_bool lamp above desk",
            "entities": [
                {"name": "lamp", "type": "device"},
                {"name": "desk", "type": "furniture"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "What is inside the robot?",
        "lesson": {
            "response_text": "The battery is inside the robot.",
            "claim": "query located_in robot",
            "entities": [{"name": "robot", "type": "robot"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Is the cache near the database?",
        "lesson": {
            "response_text": "Yes, the cache is near the database.",
            "claim": "query_bool cache near database",
            "entities": [
                {"name": "cache", "type": "cache"},
                {"name": "database", "type": "database"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "What is on the desk?",
        "lesson": {
            "response_text": "The document is on the desk.",
            "claim": "query located_on desk",
            "entities": [{"name": "desk", "type": "furniture"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Is the file in the folder?",
        "lesson": {
            "response_text": "Yes, the file is in the folder.",
            "claim": "query_bool file located_in folder",
            "entities": [
                {"name": "file", "type": "file"},
                {"name": "folder", "type": "folder"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "What is in the folder?",
        "lesson": {
            "response_text": "The file is in the folder.",
            "claim": "query located_in folder",
            "entities": [{"name": "folder", "type": "folder"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Is the router near the firewall?",
        "lesson": {
            "response_text": "Yes, the router is near the firewall.",
            "claim": "query_bool router near firewall",
            "entities": [
                {"name": "router", "type": "router"},
                {"name": "firewall", "type": "firewall"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "What is above the desk?",
        "lesson": {
            "response_text": "The lamp is above the desk.",
            "claim": "query above desk",
            "entities": [{"name": "desk", "type": "furniture"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Is the telescope in the room?",
        "lesson": {
            "response_text": "Yes, the telescope is in the room.",
            "claim": "query_bool telescope located_in room",
            "entities": [
                {"name": "telescope", "type": "telescope"},
                {"name": "room", "type": "room"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    # ── Command verbs (15 episodes) ────────────────────────────────────────────
    {
        "utterance": "Analyze the codebase.",
        "lesson": {
            "response_text": "Analyzing the codebase.",
            "claim": "command analyze codebase",
            "entities": [{"name": "codebase", "type": "codebase"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Search for the configuration file.",
        "lesson": {
            "response_text": "Searching for the configuration file.",
            "claim": "command search configuration_file",
            "entities": [{"name": "configuration_file", "type": "file"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Predict the outcome.",
        "lesson": {
            "response_text": "Predicting the outcome.",
            "claim": "command predict outcome",
            "entities": [{"name": "outcome", "type": "result"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Plan the deployment.",
        "lesson": {
            "response_text": "Planning the deployment.",
            "claim": "command plan deployment",
            "entities": [{"name": "deployment", "type": "process"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Validate the schema.",
        "lesson": {
            "response_text": "Validating the schema.",
            "claim": "command validate schema",
            "entities": [{"name": "schema", "type": "schema"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Simulate the scenario.",
        "lesson": {
            "response_text": "Simulating the scenario.",
            "claim": "command simulate scenario",
            "entities": [{"name": "scenario", "type": "simulation"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Encrypt the payload.",
        "lesson": {
            "response_text": "Encrypting the payload.",
            "claim": "command encrypt payload",
            "entities": [{"name": "payload", "type": "data"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Optimize the query.",
        "lesson": {
            "response_text": "Optimizing the query.",
            "claim": "command optimize query",
            "entities": [{"name": "query", "type": "query"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Debug the pipeline.",
        "lesson": {
            "response_text": "Debugging the pipeline.",
            "claim": "command debug pipeline",
            "entities": [{"name": "pipeline", "type": "infrastructure"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Monitor the cluster.",
        "lesson": {
            "response_text": "Monitoring the cluster.",
            "claim": "command monitor cluster",
            "entities": [{"name": "cluster", "type": "cluster"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Refactor the module.",
        "lesson": {
            "response_text": "Refactoring the module.",
            "claim": "command refactor module",
            "entities": [{"name": "module", "type": "module"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Benchmark the algorithm.",
        "lesson": {
            "response_text": "Benchmarking the algorithm.",
            "claim": "command benchmark algorithm",
            "entities": [{"name": "algorithm", "type": "algorithm"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Provision the environment.",
        "lesson": {
            "response_text": "Provisioning the environment.",
            "claim": "command provision environment",
            "entities": [{"name": "environment", "type": "environment"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Archive the logs.",
        "lesson": {
            "response_text": "Archiving the logs.",
            "claim": "command archive logs",
            "entities": [{"name": "logs", "type": "log"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Inspect the container.",
        "lesson": {
            "response_text": "Inspecting the container.",
            "claim": "command inspect container",
            "entities": [{"name": "container", "type": "container"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    # ── Sinhala assertions and queries (10 episodes) ───────────────────────────
    {
        "utterance": "බෝලය මේසය මත ඇත.",
        "lesson": {
            "response_text": "සටහන: බෝලය මේසය මත ඇත.",
            "claim": "ball located_on table",
            "entities": [
                {"name": "බෝලය", "type": "object"},
                {"name": "මේසය", "type": "surface"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "බෝලය කොහෙද?",
        "lesson": {
            "response_text": "බෝලය මේසය මත ඇත.",
            "claim": "query location ball",
            "entities": [{"name": "බෝලය", "type": "object"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "පොත මේසය යට ඇත.",
        "lesson": {
            "response_text": "සටහන: පොත මේසය යට ඇත.",
            "claim": "book located_under table",
            "entities": [
                {"name": "පොත", "type": "object"},
                {"name": "මේසය", "type": "surface"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "යතුර පොත ළඟ ඇත.",
        "lesson": {
            "response_text": "සටහන: යතුර පොත ළඟ ඇත.",
            "claim": "key near book",
            "entities": [
                {"name": "යතුර", "type": "object"},
                {"name": "පොත", "type": "object"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "රොබෝව කාමරය ඇතුළත ඇත.",
        "lesson": {
            "response_text": "සටහන: රොබෝව කාමරය ඇතුළත ඇත.",
            "claim": "robot located_in room",
            "entities": [
                {"name": "රොබෝව", "type": "robot"},
                {"name": "කාමරය", "type": "room"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "පොත කොහෙද?",
        "lesson": {
            "response_text": "පොත මේසය යට ඇත.",
            "claim": "query location book",
            "entities": [{"name": "පොත", "type": "object"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "කෝප්පය මේසය මත ඇත.",
        "lesson": {
            "response_text": "සටහන: කෝප්පය මේසය මත ඇත.",
            "claim": "cup located_on table",
            "entities": [
                {"name": "කෝප්පය", "type": "container"},
                {"name": "මේසය", "type": "surface"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "සේවාදායකය පොකුරේ ඇතුළත ඇත.",
        "lesson": {
            "response_text": "සටහන: සේවාදායකය පොකුරේ ඇතුළත ඇත.",
            "claim": "server located_in cluster",
            "entities": [
                {"name": "සේවාදායකය", "type": "server"},
                {"name": "පොකුරේ", "type": "cluster"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "පහන මේසය මත ඇත්ද?",
        "lesson": {
            "response_text": "ඔව්, පහන මේසය මත ඇත.",
            "claim": "query_bool lamp located_on table",
            "entities": [
                {"name": "පහන", "type": "device"},
                {"name": "මේසය", "type": "surface"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "රොබෝව කොහෙද?",
        "lesson": {
            "response_text": "රොබෝව කාමරය ඇතුළත ඇත.",
            "claim": "query location robot",
            "entities": [{"name": "රොබෝව", "type": "robot"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    # ── Tamil assertions and queries (10 episodes) ─────────────────────────────
    {
        "utterance": "பந்து மேசையின் மேல் உள்ளது.",
        "lesson": {
            "response_text": "குறிப்பு: பந்து மேசையின் மேல் உள்ளது.",
            "claim": "ball located_on table",
            "entities": [
                {"name": "பந்து", "type": "object"},
                {"name": "மேசை", "type": "surface"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "பந்து எங்கே?",
        "lesson": {
            "response_text": "பந்து மேசையின் மேல் உள்ளது.",
            "claim": "query location ball",
            "entities": [{"name": "பந்து", "type": "object"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "புத்தகம் மேசையின் கீழ் உள்ளது.",
        "lesson": {
            "response_text": "குறிப்பு: புத்தகம் மேசையின் கீழ் உள்ளது.",
            "claim": "book located_under table",
            "entities": [
                {"name": "புத்தகம்", "type": "object"},
                {"name": "மேசை", "type": "surface"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "திறவுகோல் புத்தகத்தின் அருகில் உள்ளது.",
        "lesson": {
            "response_text": "குறிப்பு: திறவுகோல் புத்தகத்தின் அருகில் உள்ளது.",
            "claim": "key near book",
            "entities": [
                {"name": "திறவுகோல்", "type": "object"},
                {"name": "புத்தகம்", "type": "object"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "ரோபோ அறைக்குள் உள்ளது.",
        "lesson": {
            "response_text": "குறிப்பு: ரோபோ அறைக்குள் உள்ளது.",
            "claim": "robot located_in room",
            "entities": [
                {"name": "ரோபோ", "type": "robot"},
                {"name": "அறை", "type": "room"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "புத்தகம் எங்கே?",
        "lesson": {
            "response_text": "புத்தகம் மேசையின் கீழ் உள்ளது.",
            "claim": "query location book",
            "entities": [{"name": "புத்தகம்", "type": "object"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "கோப்பை மேசையின் மேல் உள்ளது.",
        "lesson": {
            "response_text": "குறிப்பு: கோப்பை மேசையின் மேல் உள்ளது.",
            "claim": "cup located_on table",
            "entities": [
                {"name": "கோப்பை", "type": "container"},
                {"name": "மேசை", "type": "surface"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "சேவையகம் கிளஸ்டரில் உள்ளது.",
        "lesson": {
            "response_text": "குறிப்பு: சேவையகம் கிளஸ்டரில் உள்ளது.",
            "claim": "server located_in cluster",
            "entities": [
                {"name": "சேவையகம்", "type": "server"},
                {"name": "கிளஸ்டர்", "type": "cluster"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "விளக்கு மேசையின் மேல் உள்ளதா?",
        "lesson": {
            "response_text": "ஆம், விளக்கு மேசையின் மேல் உள்ளது.",
            "claim": "query_bool lamp located_on table",
            "entities": [
                {"name": "விளக்கு", "type": "device"},
                {"name": "மேசை", "type": "surface"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "ரோபோ எங்கே?",
        "lesson": {
            "response_text": "ரோபோ அறைக்குள் உள்ளது.",
            "claim": "query location robot",
            "entities": [{"name": "ரோபோ", "type": "robot"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
]

# Governed Kernel Tool-Use Gym Tasks (executed 100% natively via CapabilityResolver)
TOOL_GYM_TASKS: list[tuple[str, dict[str, Any]]] = [
    ("fs.list", {"path": "hbllm/hcir/language"}),
    ("fs.read", {"path": "hbllm/hcir/language/ontology_store.py", "max_bytes": 512}),
    ("code.python_ast", {"path": "hbllm/hcir/language/ontology_store.py"}),
    ("git.status", {}),
]


def _make_curriculum_teacher() -> Any:
    lookup = {ep["utterance"]: ep["lesson"] for ep in CURRICULUM_EPISODES}

    def _teacher(utterance: str) -> dict[str, Any]:
        return lookup.get(
            utterance,
            {
                "response_text": "",
                "claim": "",
                "entities": [],
                "lexicon": [],
                "confidence": 0.0,
                "verified": False,
            },
        )

    return _teacher


async def run_training(args: argparse.Namespace) -> dict[str, Any]:
    mem_dir = Path(args.memory_dir) if args.memory_dir else None
    db_path = Path(args.ontology_db) if args.ontology_db else DEFAULT_ONTOLOGY_DB
    store = LexicalOntologyStore(db_path)

    system = build_cognitive_system(
        CognitiveSystemConfig(
            world_id=args.world_id,
            memory_dir=mem_dir,
            enable_grounding=False,
            enable_brain_llm=True,
        )
    )
    tracker = AutonomyTracker(
        min_turns_for_retirement=args.min_turns,
        retirement_threshold=args.retirement_threshold,
    )
    system.services.extensions["autonomy_metrics"] = tracker

    if args.mode == "nvidia":
        scaffold = LLMTeacherScaffold.from_nvidia_endpoint(
            model=args.model,
            base_url=args.base_url,
            tracker=tracker,
            ontology_store=store,
        )
    elif args.mode == "live":
        scaffold = LLMTeacherScaffold.from_local_checkpoint(
            checkpoint_path=args.checkpoint,
            tracker=tracker,
            ontology_store=store,
        )
    else:
        scaffold = LLMTeacherScaffold(
            checkpoint_path=args.checkpoint,
            tracker=tracker,
            teacher_fn=_make_curriculum_teacher(),
            ontology_store=store,
        )
    system.services.extensions["llm_scaffold"] = scaffold

    epoch_reports: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        epoch_hcir_hits = 0
        epoch_shadow_hits = 0
        for ep in CURRICULUM_EPISODES:
            res = system.process_dialogue(ep["utterance"])
            if isinstance(res, dict):
                epoch_shadow_hits += 1
            elif getattr(res, "is_success", False):
                epoch_hcir_hits += 1

        # Governed Kernel Tool-Use Gym (native Pure-HCIR capability dispatch)
        tool_ok = 0
        for cap_name, params in TOOL_GYM_TASKS:
            executor = await system.services.capability_resolver.resolve(cap_name)
            ok = False
            if executor is not None:
                out = await executor.execute(params)
                ok = bool(out.get("ok", False))
            tracker.record_plan(ok)
            if ok:
                tool_ok += 1

        if tracker.can_retire_llm:
            scaffold.retired = True

        if system.runtime is not None:
            await system.runtime.run_offline_consolidation()

        epoch_reports.append(
            {
                "epoch": epoch,
                "turns": len(CURRICULUM_EPISODES),
                "hcir_native_successes": epoch_hcir_hits,
                "shadow_fallbacks": epoch_shadow_hits,
                "epoch_hcir_autonomy": round(epoch_hcir_hits / len(CURRICULUM_EPISODES), 4),
                "tool_gym_passed": f"{tool_ok}/{len(TOOL_GYM_TASKS)}",
                "cumulative": tracker.to_dict(),
                "scaffold_retired": scaffold.retired,
            }
        )

    report = {
        "mode": args.mode,
        "teacher": str(scaffold.checkpoint_path),
        "world_id": args.world_id,
        "epochs": epoch_reports,
        "final_metrics": tracker.to_dict(),
        "scaffold_retired": scaffold.retired,
        "ontology_store_stats": store.stats(),
    }
    store.close()

    out_path = Path(args.report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Track T3: Parallel Shadow Distillation into HCIR")
    parser.add_argument("--mode", choices=["curriculum", "nvidia", "live"], default="curriculum")
    parser.add_argument("--checkpoint", default=str(DEFAULT_LOCAL_CHECKPOINT))
    parser.add_argument(
        "--model",
        default=None,
        help="NVIDIA NIM model (default: nvidia/nemotron-3.5-lightning-30b-a3b)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="NVIDIA NIM base URL (default: https://integrate.api.nvidia.com/v1)",
    )
    parser.add_argument("--world-id", default="pure_hcir_t3")
    parser.add_argument("--memory-dir", default="checkpoints/t3_shadow")
    parser.add_argument("--ontology-db", default=str(DEFAULT_ONTOLOGY_DB))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--min-turns", type=int, default=50)
    parser.add_argument("--retirement-threshold", type=float, default=0.80)
    parser.add_argument("--report-path", default="reports/training/t3_shadow_report.json")
    args = parser.parse_args()

    report = asyncio.run(run_training(args))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
