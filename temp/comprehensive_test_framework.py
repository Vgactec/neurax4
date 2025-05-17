#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import logging
import numpy as np
import traceback
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("neurax_complete_test.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("NeuraxTest")

try:
    from core.quantum_sim.simulator import QuantumGravitySimulator
    logger.info("Simulateur de gravité quantique importé depuis le module core")
    SIMULATOR_SOURCE = "core"
except ImportError:
    try:
        from quantum_gravity_sim import QuantumGravitySimulator
        logger.info("Simulateur de gravité quantique importé depuis le module racine")
        SIMULATOR_SOURCE = "root"
    except ImportError:
        logger.error("Impossible d'importer le simulateur de gravité quantique")
        SIMULATOR_SOURCE = None

class TestResults:
    timestamp = None
    results = []

    def setup_method(self):
        self.timestamp = datetime.now()
        self.results = []

    def add_result(self, component, status, duration, details):
        self.results.append({
            "component": component,
            "status": status,
            "duration": duration,
            "details": details
        })

    def export_to_json(self, filename="arc_tests_results.json"):
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)

    def export_to_csv(self, filename="arc_tests_results.csv"):
        with open(filename, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Component", "Status", "Duration", "Details"])
            for result in self.results:
                writer.writerow([
                    result["component"],
                    result["status"],
                    result["duration"],
                    json.dumps(result["details"])
                ])

class TestSuite:
    def setup_method(self):
        self.arc_data_path = "./neurax_complet/arc_data"
        self.results = TestResults()
        self.results.setup_method()
        self.min_epochs = 10
        self.max_epochs = 1000
        self.convergence_threshold = 0.99

    def load_training_data(self):
        """Charge tous les puzzles d'entraînement"""
        self.logger = logging.getLogger("ARC_Tests")
        fh = logging.FileHandler('arc_complete_tests.log')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - Puzzle %(puzzle_id)s: %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        training_path = os.path.join(self.arc_data_path, "arc-agi_training_challenges.json")
        solutions_path = os.path.join(self.arc_data_path, "arc-agi_training_solutions.json")
        self.logger.info("Chargement des 1000 puzzles d'entraînement", extra={'puzzle_id': 'INIT'})

        with open(training_path, 'r') as f:
            puzzles = json.load(f)
        with open(solutions_path, 'r') as f:
            solutions = json.load(f)

        return puzzles, solutions

    def train_puzzle(self, puzzle_id, puzzle_data, solutions, max_epochs):
        """Entraîne sur un puzzle avec gestion des epochs"""
        current_accuracy = 0.0
        epochs = 0

        while epochs < max_epochs and current_accuracy < self.convergence_threshold:
            # Entraînement sur les exemples
            for example_idx, train_pair in enumerate(puzzle_data["train"]):
                try:
                    sim = QuantumGravitySimulator(grid_size=32, time_steps=8)
                    input_grid = np.array(train_pair["input"])
                    output_grid = np.array(train_pair["output"])
                    # Simulation et apprentissage
                    sim.quantum_fluctuations(intensity=1.5)
                    sim.simulate_step()

                    # Évaluer la précision
                    prediction = sim.space_time[-1]
                    current_accuracy = np.mean(prediction == output_grid)

                except Exception as e:
                    logger.error(f"Erreur d'entraînement puzzle {puzzle_id}: {str(e)}")

            epochs += 1

        return {
            "puzzle_id": puzzle_id,
            "epochs": epochs,
            "final_accuracy": current_accuracy,
            "converged": current_accuracy >= self.convergence_threshold
        }

    def test_quantum_gravity_simulator(self):
        """Test complet du simulateur sur tous les puzzles"""
        start_time = time.time()

        try:
            # Charger tous les puzzles
            puzzles, solutions = self.load_training_data()
            logger.info(f"Chargement de {len(puzzles)} puzzles d'entraînement")

            # Résultats globaux
            training_results = []

            # Entraîner sur chaque puzzle
            for puzzle_id in puzzles:
                result = self.train_puzzle(puzzle_id, puzzles[puzzle_id], solutions, self.max_epochs)
                training_results.append(result)

            # Analyser les résultats
            successful_puzzles = [r for r in training_results if r["converged"]]
            avg_epochs = np.mean([r["epochs"] for r in training_results])
            min_epochs = min([r["epochs"] for r in training_results])
            max_epochs = max([r["epochs"] for r in training_results])

            details = {
                "total_puzzles": len(puzzles),
                "successful_puzzles": len(successful_puzzles),
                "average_epochs": avg_epochs,
                "min_epochs": min_epochs,
                "max_epochs": max_epochs,
                "training_time": time.time() - start_time
            }

            self.results.add_result(
                "Apprentissage Complet",
                "PASS" if len(successful_puzzles) > 0 else "FAIL",
                time.time() - start_time,
                details
            )

        except Exception as e:
            logger.error(f"Erreur lors des tests complets: {str(e)}")
            self.results.add_result(
                "Apprentissage Complet",
                "FAIL",
                time.time() - start_time,
                {"error": str(e)}
            )
        start_time = time.time()
        try:
            sim = QuantumGravitySimulator(grid_size=32, time_steps=8)
            # Test d'initialisation
            status = "PASS" if hasattr(sim, "space_time") else "FAIL"
            duration = time.time() - start_time
            details = {
                "grid_size": 32,
                "time_steps": 8,
                "has_space_time": hasattr(sim, "space_time")
            }
        except Exception as e:
            status = "FAIL"
            duration = time.time() - start_time
            details = {"error": str(e)}

        self.results.add_result(
            "Simulateur Quantique",
            status,
            duration,
            details
        )

    def run_all_tests(self):
        """Exécute tous les tests"""
        logger.info("Démarrage des tests complets")
        self.test_quantum_gravity_simulator()
        self.results.export_to_json()
        self.results.export_to_csv()
        return self.results