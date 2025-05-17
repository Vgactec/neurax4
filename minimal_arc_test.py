#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test minimal pour Neurax2 avec les puzzles ARC
"""

import os
import sys
import json
import logging
import numpy as np
from quantum_gravity_sim_mobile import QuantumGravitySimulatorMobile

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MinimalTest")

# Configuration du test
os.makedirs("./test_data", exist_ok=True)
os.makedirs("./test_output", exist_ok=True)

# Créer un puzzle de test
test_puzzle = {
    "train": [
        {
            "input": [[0, 0], [0, 0]],
            "output": [[1, 1], [1, 1]]
        }
    ],
    "test": [
        {
            "input": [[0, 0], [0, 0]],
            "output": [[1, 1], [1, 1]]
        }
    ]
}

test_puzzle_path = "./test_data/test_puzzle.json"
with open(test_puzzle_path, 'w') as f:
    json.dump(test_puzzle, f)

logger.info("Test puzzle créé")

# Créer le simulateur
simulator = QuantumGravitySimulatorMobile(
    grid_size=16,
    time_steps=4,
    precision="float16",
    use_cache=True
)

logger.info("Simulateur créé")

# Traiter le puzzle
results = simulator.process_puzzle(test_puzzle)

# Sauvegarder les résultats
with open("./test_output/results.json", 'w') as f:
    json.dump(results, f, indent=2)

logger.info(f"Test terminé avec résultats: {results}")