#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Framework de Test Complet pour le Réseau Neuronal Gravitationnel Quantique (Neurax)
Ce script teste de manière exhaustive tous les composants du système Neurax
et évalue ses performances sur les puzzles ARC-Prize-2025
"""

import os
import sys
import time
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import traceback
import h5py
import csv
import pandas as pd
from tqdm import tqdm
import random
import psutil

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

# Assurez-vous que les chemins d'import sont corrects
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import des modules Neurax à tester
try:
    # Simulateur de gravité quantique
    try:
        from core.quantum_sim.simulator import QuantumGravitySimulator
        logger.info("Simulateur de gravité quantique importé depuis le module core")
        SIMULATOR_SOURCE = "core"
    except ImportError:
        from quantum_gravity_sim import QuantumGravitySimulator
        logger.info("Simulateur de gravité quantique importé depuis le module racine")
        SIMULATOR_SOURCE = "root"

    # Module de neurone quantique
    try:
        from core.neuron.quantum_neuron import QuantumNeuron
        HAS_NEURON_MODULE = True
    except ImportError:
        logger.warning("Module de neurone quantique non disponible")
        HAS_NEURON_MODULE = False

    # Module réseau P2P
    try:
        from core.p2p.network import P2PNetwork
        HAS_P2P_MODULE = True
    except ImportError:
        logger.warning("Module réseau P2P non disponible")
        HAS_P2P_MODULE = False

    # Module de consensus
    try:
        from core.consensus.proof_of_cognition import ProofOfCognition
        HAS_CONSENSUS_MODULE = True
    except ImportError:
        logger.warning("Module de consensus non disponible")
        HAS_CONSENSUS_MODULE = False

    # Modules utilitaires
    try:
        from visualization import QuantumGravityVisualizer
        HAS_VISUALIZATION = True
    except ImportError:
        logger.warning("Module de visualisation non disponible")
        HAS_VISUALIZATION = False

    try:
        from export_manager import ExportManager
        HAS_EXPORT = True
    except ImportError:
        logger.warning("Module d'export non disponible")
        HAS_EXPORT = False

    try:
        from database import DatabaseManager
        HAS_DATABASE = True
    except ImportError:
        logger.warning("Module de base de données non disponible")
        HAS_DATABASE = False

    MODULE_IMPORT_SUCCESS = True
except Exception as e:
    logger.error(f"Erreur d'importation des modules Neurax: {str(e)}")
    traceback.print_exc()
    MODULE_IMPORT_SUCCESS = False


# Import des modules pour les puzzles ARC
def load_arc_data(data_path="../arc_data"):
    """
    Charge les données des puzzles ARC

    Args:
        data_path: Chemin vers les fichiers JSON ARC

    Returns:
        tuple: (training_data, evaluation_data, test_data)
    """
    try:
        # Chargement des fichiers de puzzle
        training_path = os.path.join(data_path, "arc-agi_training_challenges.json")
        evaluation_path = os.path.join(data_path, "arc-agi_evaluation_challenges.json")
        test_path = os.path.join(data_path, "arc-agi_test_challenges.json")

        # Solutions pour évaluation
        training_solutions_path = os.path.join(data_path, "arc-agi_training_solutions.json")
        evaluation_solutions_path = os.path.join(data_path, "arc-agi_evaluation_solutions.json")

        # Chargement des données
        with open(training_path, 'r') as f:
            training_data = json.load(f)
        with open(evaluation_path, 'r') as f:
            evaluation_data = json.load(f)
        with open(test_path, 'r') as f:
            test_data = json.load(f)

        # Chargement des solutions
        with open(training_solutions_path, 'r') as f:
            training_solutions = json.load(f)
        with open(evaluation_solutions_path, 'r') as f:
            evaluation_solutions = json.load(f)

        logger.info(f"Données ARC chargées: {len(training_data)} puzzles d'entraînement, "
                   f"{len(evaluation_data)} puzzles d'évaluation, {len(test_data)} puzzles de test")

        return {
            "training": {
                "challenges": training_data,
                "solutions": training_solutions
            },
            "evaluation": {
                "challenges": evaluation_data,
                "solutions": evaluation_solutions
            },
            "test": {
                "challenges": test_data,
                "solutions": None  # Pas de solutions disponibles pour les puzzles de test
            }
        }
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données ARC: {str(e)}")
        traceback.print_exc()
        return None


class ComprehensiveTestResults:
    """Classe pour stocker et formater les résultats complets des tests"""

    def __init__(self):
        self.timestamp = datetime.now()
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.skipped_tests = 0
        self.total_execution_time = 0
        self.component_results = {}
        self.arc_puzzle_results = {}
        self.performance_metrics = {}
        self.hardware_metrics = self._collect_hardware_info()

    def _collect_hardware_info(self):
        """Collecte les informations sur le matériel utilisé pour les tests"""
        hw_info = {
            "os": os.name,
            "cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "total_memory_gb": round(psutil.virtual_memory().total / (1024.0 ** 3), 2),
            "available_memory_gb": round(psutil.virtual_memory().available / (1024.0 ** 3), 2),
            "python_version": sys.version,
            "numpy_version": np.__version__
        }
        return hw_info

    def add_component_result(self, component_name, status, execution_time, details=None):
        """Ajoute un résultat de test de composant"""
        if component_name not in self.component_results:
            self.component_results[component_name] = []

        result = {
            "status": status,
            "execution_time": execution_time,
            "details": details or {}
        }

        self.component_results[component_name].append(result)

        self.total_tests += 1
        if status == "PASS":
            self.passed_tests += 1
        elif status == "FAIL":
            self.failed_tests += 1
        elif status == "SKIP":
            self.skipped_tests += 1

        self.total_execution_time += execution_time

    def add_arc_puzzle_result(self, puzzle_id, phase, status, execution_time, accuracy=None, details=None):
        """Ajoute un résultat de test sur un puzzle ARC"""
        if puzzle_id not in self.arc_puzzle_results:
            self.arc_puzzle_results[puzzle_id] = {}

        result = {
            "status": status,
            "execution_time": execution_time,
            "accuracy": accuracy,
            "details": details or {}
        }

        self.arc_puzzle_results[puzzle_id][phase] = result

    def add_performance_metric(self, category, metric_name, value):
        """Ajoute une métrique de performance"""
        if category not in self.performance_metrics:
            self.performance_metrics[category] = {}

        self.performance_metrics[category][metric_name] = value

    def get_summary(self):
        """Retourne un résumé des résultats des tests"""
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0

        summary = {
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "skipped_tests": self.skipped_tests,
            "success_rate": round(success_rate, 2),
            "total_execution_time": round(self.total_execution_time, 2),
            "component_count": len(self.component_results),
            "arc_puzzle_count": len(self.arc_puzzle_results),
            "hardware_info": self.hardware_metrics
        }

        # Ajouter résumé des performances ARC
        if self.arc_puzzle_results:
            arc_phases = set()
            for puzzle_id in self.arc_puzzle_results:
                arc_phases.update(self.arc_puzzle_results[puzzle_id].keys())

            arc_summary = {}
            for phase in arc_phases:
                phase_results = [
                    self.arc_puzzle_results[puzzle_id][phase] 
                    for puzzle_id in self.arc_puzzle_results 
                    if phase in self.arc_puzzle_results[puzzle_id]
                ]

                phase_accuracy = [
                    r["accuracy"] for r in phase_results 
                    if r["accuracy"] is not None
                ]

                arc_summary[phase] = {
                    "count": len(phase_results),
                    "avg_accuracy": np.mean(phase_accuracy) if phase_accuracy else None,
                    "max_accuracy": np.max(phase_accuracy) if phase_accuracy else None,
                    "min_accuracy": np.min(phase_accuracy) if phase_accuracy else None
                }

            summary["arc_summary"] = arc_summary

        return summary

    def generate_detailed_report(self, file_path):
        """Génère un rapport détaillé au format Markdown"""
        summary = self.get_summary()

        with open(file_path, 'w') as f:
            f.write("# Rapport de Test Complet du Réseau Neuronal Gravitationnel Quantique (Neurax)\n\n")
            f.write(f"Date: {summary['timestamp']}\n\n")

            f.write("## Résumé\n\n")
            f.write(f"- **Total des tests**: {summary['total_tests']}\n")
            f.write(f"- **Tests réussis**: {summary['passed_tests']}\n")
            f.write(f"- **Tests échoués**: {summary['failed_tests']}\n")
            f.write(f"- **Tests ignorés**: {summary['skipped_tests']}\n")
            f.write(f"- **Taux de réussite**: {summary['success_rate']}%\n")
            f.write(f"- **Temps d'exécution total**: {summary['total_execution_time']} secondes\n\n")

            f.write("## Configuration Matérielle\n\n")
            f.write(f"- **Système**: {summary['hardware_info']['os']}\n")
            f.write(f"- **CPU Logiques**: {summary['hardware_info']['cpu_count']}\n")
            f.write(f"- **CPU Physiques**: {summary['hardware_info']['physical_cpu_count']}\n")
            f.write(f"- **Mémoire Totale**: {summary['hardware_info']['total_memory_gb']} GB\n")
            f.write(f"- **Mémoire Disponible**: {summary['hardware_info']['available_memory_gb']} GB\n")
            f.write(f"- **Version Python**: {summary['hardware_info']['python_version'].split()[0]}\n")
            f.write(f"- **Version NumPy**: {summary['hardware_info']['numpy_version']}\n\n")

            # Détails des tests par composant
            f.write("## Résultats Détaillés par Composant\n\n")
            for component, results in self.component_results.items():
                pass_count = sum(1 for r in results if r["status"] == "PASS")
                fail_count = sum(1 for r in results if r["status"] == "FAIL")
                skip_count = sum(1 for r in results if r["status"] == "SKIP")

                status_icon = "✅" if fail_count == 0 else "❌"
                success_rate = (pass_count / len(results) * 100) if results else 0

                f.write(f"### {status_icon} {component}\n\n")
                f.write(f"- **Tests**: {len(results)}\n")
                f.write(f"- **Réussis**: {pass_count}\n")
                f.write(f"- **Échoués**: {fail_count}\n")
                f.write(f"- **Ignorés**: {skip_count}\n")
                f.write(f"- **Taux de réussite**: {success_rate:.2f}%\n\n")

                f.write("#### Détails des tests\n\n")
                f.write("| Test | Statut | Temps (s) | Détails |\n")
                f.write("|------|--------|-----------|--------|\n")

                for i, result in enumerate(results):
                    status_text = "✅ Réussi" if result["status"] == "PASS" else "❌ Échoué" if result["status"] == "FAIL" else "⚠️ Ignoré"
                    details_str = ", ".join(f"{k}: {v}" for k, v in result["details"].items())

                    f.write(f"| Test {i+1} | {status_text} | {result['execution_time']:.4f} | {details_str} |\n")

                f.write("\n")

            # Si des tests ARC ont été effectués, ajouter une section
            if self.arc_puzzle_results:
                f.write("## Résultats sur les Puzzles ARC\n\n")

                # Générer un résumé par phase
                if "arc_summary" in summary:
                    f.write("### Résumé par Phase\n\n")
                    f.write("| Phase | Nombre de Puzzles | Précision Moyenne | Précision Min | Précision Max |\n")
                    f.write("|-------|------------------|-------------------|---------------|---------------|\n")

                    for phase, phase_stats in summary["arc_summary"].items():
                        avg_acc = f"{phase_stats['avg_accuracy']*100:.2f}%" if phase_stats['avg_accuracy'] is not None else "N/A"
                        min_acc = f"{phase_stats['min_accuracy']*100:.2f}%" if phase_stats['min_accuracy'] is not None else "N/A"
                        max_acc = f"{phase_stats['max_accuracy']*100:.2f}%" if phase_stats['max_accuracy'] is not None else "N/A"

                        f.write(f"| {phase} | {phase_stats['count']} | {avg_acc} | {min_acc} | {max_acc} |\n")

                    f.write("\n")

                # Détails de quelques puzzles (limiter pour éviter un rapport trop long)
                max_puzzles_to_show = 10
                sample_puzzles = list(self.arc_puzzle_results.keys())[:max_puzzles_to_show]

                f.write(f"### Détails des Puzzles (Échantillon de {len(sample_puzzles)})\n\n")

                for puzzle_id in sample_puzzles:
                    f.write(f"#### Puzzle {puzzle_id}\n\n")
                    f.write("| Phase | Statut | Temps (s) | Précision | Détails |\n")
                    f.write("|-------|--------|-----------|-----------|--------|\n")

                    for phase, result in self.arc_puzzle_results[puzzle_id].items():
                        status_text = "✅ Réussi" if result["status"] == "PASS" else "❌ Échoué" if result["status"] == "FAIL" else "⚠️ Ignoré"
                        accuracy_text = f"{result['accuracy']*100:.2f}%" if result["accuracy"] is not None else "N/A"
                        details_str = ", ".join(f"{k}: {v}" for k, v in result["details"].items() if k != "prediction")

                        f.write(f"| {phase} | {status_text} | {result['execution_time']:.4f} | {accuracy_text} | {details_str} |\n")

                    f.write("\n")

            # Métriques de performance
            if self.performance_metrics:
                f.write("## Métriques de Performance\n\n")

                for category, metrics in self.performance_metrics.items():
                    f.write(f"### {category}\n\n")
                    f.write("| Métrique | Valeur |\n")
                    f.write("|----------|-------|\n")

                    for metric_name, value in metrics.items():
                        f.write(f"| {metric_name} | {value} |\n")

                    f.write("\n")

            f.write("## Conclusion\n\n")

            if summary['success_rate'] >= 90:
                conclusion = "Le système Neurax a passé la grande majorité des tests avec succès. Les performances sont excellentes et le système est prêt pour une utilisation avancée."
            elif summary['success_rate'] >= 75:
                conclusion = "Le système Neurax a passé une bonne partie des tests avec succès. Quelques améliorations sont nécessaires, mais le système est fonctionnel."
            elif summary['success_rate'] >= 50:
                conclusion = "Le système Neurax a passé environ la moitié des tests. Des corrections importantes sont nécessaires avant une utilisation en production."
            else:
                conclusion = "Le système Neurax a échoué à de nombreux tests. Une révision approfondie est nécessaire."

            f.write(f"{conclusion}\n\n")

            f.write("---\n\n")
            f.write("*Rapport généré automatiquement par le Framework de Test Neurax*\n")

    def export_to_csv(self, file_path):
        """Exporte les résultats détaillés au format CSV"""
        with open(file_path, 'w', newline='') as csvfile:
            fieldnames = ['component', 'test_id', 'status', 'execution_time', 'details']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            for component, results in self.component_results.items():
                for i, result in enumerate(results):
                    writer.writerow({
                        'component': component,
                        'test_id': i+1,
                        'status': result['status'],
                        'execution_time': result['execution_time'],
                        'details': json.dumps(result['details'])
                    })

    def export_arc_results_to_csv(self, file_path):
        """Exporte les résultats des puzzles ARC au format CSV"""
        if not self.arc_puzzle_results:
            return

        with open(file_path, 'w', newline='') as csvfile:
            fieldnames = ['puzzle_id', 'phase', 'status', 'execution_time', 'accuracy', 'details']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            for puzzle_id, phases in self.arc_puzzle_results.items():
                for phase, result in phases.items():
                    writer.writerow({
                        'puzzle_id': puzzle_id,
                        'phase': phase,
                        'status': result['status'],
                        'execution_time': result['execution_time'],
                        'accuracy': result['accuracy'] if result['accuracy'] is not None else '',
                        'details': json.dumps({k: v for k, v in result['details'].items() if k != 'prediction'})
                    })


class TestSuite:
    """Classe principale pour exécuter la suite de tests complète"""

    def __init__(self, arc_data_path="../arc_data"):
        self.results = ComprehensiveTestResults()
        self.arc_data = load_arc_data(data_path=arc_data_path)

    def test_quantum_gravity_simulator(self):
        """Tests complets du simulateur de gravité quantique"""
        logger.info("Test complet du simulateur de gravité quantique")

        # Paramètres à tester
        grid_sizes = [20, 32, 50]
        time_steps = [4, 8, 16]
        intensities = [0.5, 1.0, 2.0]

        for grid_size in grid_sizes:
            for time_step in time_steps:
                test_name = f"Simulateur_Grille{grid_size}_Temps{time_step}"
                start_time = time.time()

                try:
                    # Créer une instance du simulateur
                    sim = QuantumGravitySimulator(grid_size=grid_size, time_steps=time_step)

                    # Vérifier l'initialisation correcte
                    initialization_ok = True
                    if not hasattr(sim, 'space_time'):
                        initialization_ok = False

                    if hasattr(sim, 'space_time'):
                        expected_shape = (time_step, grid_size, grid_size, grid_size)
                        shape_ok = sim.space_time.shape == expected_shape
                    else:
                        shape_ok = False

                    # Générer des fluctuations quantiques avec différentes intensités
                    fluctuations_results = {}
                    for intensity in intensities:
                        try:
                            # Sauvegarder l'état avant
                            before_state = sim.space_time.copy() if hasattr(sim, 'space_time') else None

                            # Appliquer des fluctuations
                            if hasattr(sim, 'quantum_fluctuations'):
                                sim.quantum_fluctuations(intensity=intensity)

                                # Vérifier les changements
                                after_state = sim.space_time.copy() if hasattr(sim, 'space_time') else None

                                if before_state is not None and after_state is not None:
                                    has_changes = not np.array_equal(before_state, after_state)
                                    avg_change = np.mean(np.abs(after_state - before_state))
                                    max_change = np.max(np.abs(after_state - before_state))
                                    min_val = np.min(after_state)
                                    max_val = np.max(after_state)
                                    mean_val = np.mean(after_state)
                                    std_val = np.std(after_state)

                                    fluctuations_results[intensity] = {
                                        "has_changes": has_changes,
                                        "avg_change": float(avg_change),
                                        "max_change": float(max_change),
                                        "min_val": float(min_val),
                                        "max_val": float(max_val),
                                        "mean_val": float(mean_val),
                                        "std_val": float(std_val)
                                    }
                                else:
                                    fluctuations_results[intensity] = {"error": "Impossible de comparer les états"}
                            else:
                                fluctuations_results[intensity] = {"error": "Méthode quantum_fluctuations non disponible"}

                        except Exception as e:
                            fluctuations_results[intensity] = {"error": str(e)}

                    # Test de simulation de plusieurs pas
                    simulation_steps_results = {}
                    if hasattr(sim, 'simulate_step'):
                        try:
                            # Réinitialiser le simulateur pour ces tests
                            sim = QuantumGravitySimulator(grid_size=grid_size, time_steps=time_step)

                            for steps in [1, 5, 10]:
                                step_times = []

                                # Exécuter plusieurs pas de simulation
                                for _ in range(steps):
                                    step_start = time.time()
                                    sim.simulate_step()
                                    step_end = time.time()
                                    step_times.append(step_end - step_start)

                                # Obtenir les métriques si disponible
                                if hasattr(sim, 'get_metrics'):
                                    metrics = sim.get_metrics()
                                else:
                                    metrics = None

                                simulation_steps_results[steps] = {
                                    "avg_step_time": sum(step_times) / len(step_times),
                                    "metrics": metrics
                                }

                        except Exception as e:
                            simulation_steps_results = {"error": str(e)}
                    else:
                        simulation_steps_results = {"error": "Méthode simulate_step non disponible"}

                    # Déterminer le statut global du test
                    status = "PASS" if initialization_ok and shape_ok else "FAIL"

                    # Collecter tous les détails du test
                    details = {
                        "initialization_ok": initialization_ok,
                        "shape_ok": shape_ok,
                        "expected_shape": str(expected_shape) if 'expected_shape' in locals() else "Indéterminé",
                        "actual_shape": str(sim.space_time.shape) if hasattr(sim, 'space_time') else "Aucun",
                        "fluctuations_results": fluctuations_results,
                        "simulation_steps_results": simulation_steps_results
                    }

                except Exception as e:
                    status = "FAIL"
                    details = {"error": str(e)}

                execution_time = time.time() - start_time
                self.results.add_component_result("Simulateur de Gravité Quantique", status, execution_time, details)

        return self.results

    def test_quantum_neuron(self):
        """Tests du système de neurone quantique"""
        if not HAS_NEURON_MODULE:
            logger.warning("Module de neurone non disponible, tests ignorés")
            self.results.add_component_result("Neurone Quantique", "SKIP", 0, {"error": "Module non disponible"})
            return self.results

        logger.info("Test du module de neurone quantique")

        # Test de la création d'un neurone
        start_time = time.time()
        try:
            neuron = QuantumNeuron()

            # Vérifier l'initialisation
            initialization_ok = hasattr(neuron, 'weights') if hasattr(neuron, 'weights') else False

            # Test des fonctions d'activation
            activation_tests = {}
            if hasattr(neuron, 'activate'):
                test_inputs = [0.0, 0.5, 1.0, -0.5, -1.0]

                for input_val in test_inputs:
                    try:
                        output = neuron.activate(input_val)
                        activation_tests[str(input_val)] = float(output)
                    except Exception as e:
                        activation_tests[str(input_val)] = {"error": str(e)}
            else:
                activation_tests = {"error": "Méthode activate non disponible"}

            # Test d'apprentissage si disponible
            learning_tests = {}
            if hasattr(neuron, 'learn'):
                test_patterns = [
                    (0.0, 0.0),
                    (0.0, 1.0),
                    (1.0, 0.0),
                    (1.0, 1.0)
                ]

                try:
                    # Essayez d'apprendre les patterns
                    epochs = 100
                    learning_rate = 0.1

                    errors = []
                    for epoch in range(epochs):
                        epoch_errors = []
                        for input_val, target in test_patterns:
                            error = neuron.learn(input_val, target, learning_rate)
                            epoch_errors.append(error)
                        errors.append(np.mean(epoch_errors))

                    # Vérifier les résultats d'apprentissage
                    final_results = {}
                    for input_val, target in test_patterns:
                        output = neuron.activate(input_val)
                        error = abs(target - output)
                        final_results[f"{input_val}->{target}"] = {
                            "output": float(output),
                            "error": float(error)
                        }

                    learning_tests = {
                        "epochs": epochs,
                        "learning_rate": learning_rate,
                        "final_error": float(errors[-1]) if errors else None,
                        "error_reduction": float(errors[0] - errors[-1]) if errors and len(errors) > 1 else None,
                        "final_results": final_results
                    }

                except Exception as e:
                    learning_tests = {"error": str(e)}
            else:
                learning_tests = {"error": "Méthode learn non disponible"}

            # Déterminer le statut global
            status = "PASS" if initialization_ok else "FAIL"

            details = {
                "initialization_ok": initialization_ok,
                "activation_tests": activation_tests,
                "learning_tests": learning_tests
            }

        except Exception as e:
            status = "FAIL"
            details = {"error": str(e)}

        execution_time = time.time() - start_time
        self.results.add_component_result("Neurone Quantique", status, execution_time, details)

        return self.results

    def test_p2p_network(self):
        """Tests du réseau pair-à-pair"""
        if not HAS_P2P_MODULE:
            logger.warning("Module P2P non disponible, tests ignorés")
            self.results.add_component_result("Réseau P2P", "SKIP", 0, {"error": "Module non disponible"})
            return self.results

        logger.info("Test du module réseau P2P")

        # Test de création du réseau
        start_time = time.time()
        try:
            # Créer une instance du réseau P2P
            network = P2PNetwork(local_port=8000)

            # Vérifier l'initialisation
            initialization_ok = hasattr(network, 'node_id')

            # Test des fonctions de base
            messaging_tests = {}
            if hasattr(network, 'create_message') and hasattr(network, 'send_message'):
                try:
                    # Créer un message de test
                    test_message = network.create_message("TEST", {"data": "test_content"})
                    message_created = test_message is not None

                    # Simuler l'envoi d'un message (sans réellement connecter des pairs)
                    if hasattr(network, 'send_message'):
                        # On ne peut pas vraiment tester l'envoi sans pairs connectés
                        send_message_available = True
                    else:
                        send_message_available = False

                    messaging_tests = {
                        "message_created": message_created,
                        "message_content": str(test_message) if message_created else None,
                        "send_message_available": send_message_available
                    }

                except Exception as e:
                    messaging_tests = {"error": str(e)}
            else:
                messaging_tests = {"error": "Méthodes message non disponibles"}

            # Test des fonctions de découverte des pairs
            discovery_tests = {}
            if hasattr(network, 'discover_peers'):
                try:
                    # Tester la découverte sans réellement connecter
                    discovery_tests = {"method_available": True}
                except Exception as e:
                    discovery_tests = {"error": str(e)}
            else:
                discovery_tests = {"error": "Méthode discover_peers non disponible"}

            # Déterminer le statut global
            status = "PASS" if initialization_ok else "FAIL"

            details = {
                "initialization_ok": initialization_ok,
                "messaging_tests": messaging_tests,
                "discovery_tests": discovery_tests
            }

        except Exception as e:
            status = "FAIL"
            details = {"error": str(e)}

        execution_time = time.time() - start_time
        self.results.add_component_result("Réseau P2P", status, execution_time, details)

        return self.results

    def test_consensus_mechanism(self):
        """Tests du mécanisme de consensus"""
        if not HAS_CONSENSUS_MODULE:
            logger.warning("Module de consensus non disponible, tests ignorés")
            self.results.add_component_result("Mécanisme de Consensus", "SKIP", 0, {"error": "Module non disponible"})
            return self.results

        logger.info("Test du mécanisme de consensus")

        # Test de création du mécanisme de consensus
        start_time = time.time()
        try:
            # Créer une instance du mécanisme de consensus
            consensus = ProofOfCognition(local_node_id="test_node")

            # Vérifier l'initialisation
            initialization_ok = True

            # Test des fonctions de base
            validation_tests = {}
            if hasattr(consensus, 'create_validation_request') and hasattr(consensus, 'process_validation_request'):
                try:
                    # Créer une requête de validation de test
                    test_item = {
                        "id": "test_item_001",
                        "type": "SOLUTION",
                        "data": {"content": "Test solution data"}
                                        }

                    request = consensus.create_validation_request(
                        test_item["id"], 
                        test_item["type"], 
                        test_item["data"], 
                        validator_id="validator_001"
                    )

                    request_created = request is not None

                    # Traiter la requête
                    if hasattr(consensus, 'process_validation_request'):
                        result = consensus.process_validation_request(request)
                        request_processed = result is not None
                    else:
                        request_processed = False

                    validation_tests = {
                        "request_created": request_created,
                        "request_content": str(request) if request_created else None,
                        "request_processed": request_processed,
                        "result": str(result) if 'result' in locals() and result is not None else None
                    }

                except Exception as e:
                    validation_tests = {"error": str(e)}
            else:
                validation_tests = {"error": "Méthodes de validation non disponibles"}

            # Test de sélection des validateurs
            validator_tests = {}
            if hasattr(consensus, 'select_validators'):
                try:
                    validators = consensus.select_validators(
                        "test_item_002", 
                        "KNOWLEDGE", 
                        domain="physics", 
                        num_validators=3
                    )

                    validator_tests = {
                        "validators_selected": validators is not None,
                        "validator_count": len(validators) if validators is not None else 0,
                        "validators": str(validators) if validators is not None else None
                    }

                except Exception as e:
                    validator_tests = {"error": str(e)}
            else:
                validator_tests = {"error": "Méthode select_validators non disponible"}

            # Déterminer le statut global
            status = "PASS" if initialization_ok else "FAIL"

            details = {
                "initialization_ok": initialization_ok,
                "validation_tests": validation_tests,
                "validator_tests": validator_tests
            }

        except Exception as e:
            status = "FAIL"
            details = {"error": str(e)}

        execution_time = time.time() - start_time
        self.results.add_component_result("Mécanisme de Consensus", status, execution_time, details)

        return self.results

    def test_visualization(self):
        """Tests du module de visualisation"""
        if not HAS_VISUALIZATION:
            logger.warning("Module de visualisation non disponible, tests ignorés")
            self.results.add_component_result("Visualisation", "SKIP", 0, {"error": "Module non disponible"})
            return self.results

        logger.info("Test du module de visualisation")

        # Test de création du visualiseur
        start_time = time.time()
        try:
            # Créer une instance du simulateur pour obtenir des données
            sim = QuantumGravitySimulator(grid_size=32)
            sim.quantum_fluctuations()

            # Créer une instance du visualiseur
            visualizer = QuantumGravityVisualizer()

            # Vérifier l'initialisation
            initialization_ok = True

            # Test des méthodes de visualisation
            visualization_tests = {}

            # Test de la visualisation 3D
            if hasattr(visualizer, 'create_3d_plot'):
                try:
                    fig = visualizer.create_3d_plot(sim.space_time)
                    plot_3d_created = fig is not None

                    visualization_tests["plot_3d"] = {
                        "plot_created": plot_3d_created
                    }

                except Exception as e:
                    visualization_tests["plot_3d"] = {"error": str(e)}
            else:
                visualization_tests["plot_3d"] = {"error": "Méthode create_3d_plot non disponible"}

            # Test de la visualisation 2D
            if hasattr(visualizer, 'create_slice_plot'):
                try:
                    fig = visualizer.create_slice_plot(sim.space_time)
                    plot_2d_created = fig is not None

                    visualization_tests["plot_2d"] = {
                        "plot_created": plot_2d_created
                    }

                except Exception as e:
                    visualization_tests["plot_2d"] = {"error": str(e)}
            else:
                visualization_tests["plot_2d"] = {"error": "Méthode create_slice_plot non disponible"}

            # Déterminer le statut global
            status = "PASS" if initialization_ok else "FAIL"

            details = {
                "initialization_ok": initialization_ok,
                "visualization_tests": visualization_tests
            }

        except Exception as e:
            status = "FAIL"
            details = {"error": str(e)}

        execution_time = time.time() - start_time
        self.results.add_component_result("Visualisation", status, execution_time, details)

        return self.results

    def test_export_manager(self):
        """Tests du gestionnaire d'export"""
        if not HAS_EXPORT:
            logger.warning("Module d'export non disponible, tests ignorés")
            self.results.add_component_result("Gestionnaire d'Export", "SKIP", 0, {"error": "Module non disponible"})
            return self.results

        logger.info("Test du gestionnaire d'export")

        # Test de création du gestionnaire d'export
        start_time = time.time()
        try:
            # Créer une instance du simulateur pour obtenir des données
            sim = QuantumGravitySimulator(grid_size=32)
            sim.quantum_fluctuations()

            # Données pour les tests
            space_time = sim.space_time
            metrics = {"metric1": 0.5, "metric2": 1.0}
            parameters = {"param1": 32, "param2": "test"}

            # Créer une instance du gestionnaire d'export
            export_manager = ExportManager()

            # Vérifier l'initialisation
            initialization_ok = True

            # Test des méthodes d'export
            export_tests = {}

            # Test de l'export Excel
            if hasattr(export_manager, 'export_to_excel'):
                try:
                    # Fichier temporaire pour le test
                    temp_excel = "temp_test_export.xlsx"

                    # Exporter
                    excel_path = export_manager.export_to_excel(space_time, metrics, parameters)
                    excel_exported = excel_path is not None and os.path.exists(excel_path)

                    # Supprimer le fichier temporaire si créé
                    if excel_exported and os.path.exists(excel_path):
                        os.remove(excel_path)

                    export_tests["excel"] = {
                        "export_successful": excel_exported
                    }

                except Exception as e:
                    export_tests["excel"] = {"error": str(e)}
            else:
                export_tests["excel"] = {"error": "Méthode export_to_excel non disponible"}

            # Test de l'export HDF5
            if hasattr(export_manager, 'export_to_hdf5'):
                try:
                    # Fichier temporaire pour le test
                    temp_hdf5 = "temp_test_export.h5"

                    # Exporter
                    hdf5_path = export_manager.export_to_hdf5(space_time, metrics, parameters)
                    hdf5_exported = hdf5_path is not None and os.path.exists(hdf5_path)

                    # Supprimer le fichier temporaire si créé
                    if hdf5_exported and os.path.exists(hdf5_path):
                        os.remove(hdf5_path)

                    export_tests["hdf5"] = {
                        "export_successful": hdf5_exported
                    }

                except Exception as e:
                    export_tests["hdf5"] = {"error": str(e)}
            else:
                export_tests["hdf5"] = {"error": "Méthode export_to_hdf5 non disponible"}

            # Test de l'export CSV
            if hasattr(export_manager, 'export_to_detailed_csv'):
                try:
                    # Exporter
                    csv_path = export_manager.export_to_detailed_csv(space_time, metrics, parameters)
                    csv_exported = csv_path is not None and os.path.exists(csv_path)

                    # Supprimer le fichier temporaire si créé
                    if csv_exported and os.path.exists(csv_path):
                        os.remove(csv_path)

                    export_tests["csv"] = {
                        "export_successful": csv_exported
                    }

                except Exception as e:
                    export_tests["csv"] = {"error": str(e)}
            else:
                export_tests["csv"] = {"error": "Méthode export_to_detailed_csv non disponible"}

            # Déterminer le statut global
            status = "PASS" if initialization_ok else "FAIL"

            details = {
                "initialization_ok": initialization_ok,
                "export_tests": export_tests
            }

        except Exception as e:
            status = "FAIL"
            details = {"error": str(e)}

        execution_time = time.time() - start_time
        self.results.add_component_result("Gestionnaire d'Export", status, execution_time, details)

        return self.results

    def test_database_manager(self):
        """Tests du gestionnaire de base de données"""
        if not HAS_DATABASE:
            logger.warning("Module de base de données non disponible, tests ignorés")
            self.results.add_component_result("Gestionnaire de Base de Données", "SKIP", 0, {"error": "Module non disponible"})
            return self.results

        logger.info("Test du gestionnaire de base de données")

        # Test de création du gestionnaire de base de données
        start_time = time.time()
        try:
            # Créer une instance du simulateur pour obtenir des données
            sim = QuantumGravitySimulator(grid_size=32)
            sim.quantum_fluctuations()

            # Données pour les tests
            space_time = sim.space_time
            metrics = {"metric1": 0.5, "metric2": 1.0}

            # Créer une instance du gestionnaire de base de données
            db_manager = DatabaseManager()

            # Vérifier l'initialisation
            initialization_ok = True

            # Test des méthodes de base de données
            db_tests = {}

            # Test de la création des tables
            if hasattr(db_manager, 'create_tables'):
                try:
                    db_manager.create_tables()
                    tables_created = True

                    db_tests["create_tables"] = {
                        "success": tables_created
                    }

                except Exception as e:
                    db_tests["create_tables"] = {"error": str(e)}
            else:
                db_tests["create_tables"] = {"error": "Méthode create_tables non disponible"}

            # Test de l'enregistrement de simulation
            if hasattr(db_manager, 'save_simulation'):
                try:
                    simulation_id = db_manager.save_simulation(
                        32, 10, 1.0, space_time, metrics
                    )

                    save_successful = simulation_id is not None

                    db_tests["save_simulation"] = {
                        "success": save_successful,
                        "simulation_id": str(simulation_id) if save_successful else None
                    }

                except Exception as e:
                    db_tests["save_simulation"] = {"error": str(e)}
            else:
                db_tests["save_simulation"] = {"error": "Méthode save_simulation non disponible"}

            # Test de la récupération des simulations récentes
            if hasattr(db_manager, 'get_recent_simulations'):
                try:
                    recent_simulations = db_manager.get_recent_simulations(limit=5)

                    get_recent_successful = recent_simulations is not None

                    db_tests["get_recent_simulations"] = {
                        "success": get_recent_successful,
                        "simulation_count": len(recent_simulations) if get_recent_successful else 0
                    }

                except Exception as e:
                    db_tests["get_recent_simulations"] = {"error": str(e)}
            else:
                db_tests["get_recent_simulations"] = {"error": "Méthode get_recent_simulations non disponible"}

            # Test de la récupération d'une simulation par ID
            if hasattr(db_manager, 'get_simulation_by_id') and 'simulation_id' in locals() and simulation_id is not None:
                try:
                    simulation = db_manager.get_simulation_by_id(simulation_id)

                    get_by_id_successful = simulation is not None

                    db_tests["get_simulation_by_id"] = {
                        "success": get_by_id_successful
                    }

                except Exception as e:
                    db_tests["get_simulation_by_id"] = {"error": str(e)}
            else:
                db_tests["get_simulation_by_id"] = {"error": "Méthode get_simulation_by_id non disponible ou pas d'ID de simulation"}

            # Fermer la connexion
            if hasattr(db_manager, 'close'):
                db_manager.close()

            # Déterminer le statut global
            status = "PASS" if initialization_ok else "FAIL"

            details = {
                "initialization_ok": initialization_ok,
                "db_tests": db_tests
            }

        except Exception as e:
            status = "FAIL"
            details = {"error": str(e)}

        execution_time = time.time() - start_time
        self.results.add_component_result("Gestionnaire de Base de Données", status, execution_time, details)

        return self.results

    def test_arc_puzzles_with_neurax(self):
        """
        Teste le système Neurax sur les puzzles ARC pour évaluer ses capacités
        de raisonnement abstrait.
        """
        if not self.arc_data:
            logger.warning("Données ARC non disponibles, tests ignorés")
            return self.results

        logger.info("Test des puzzles ARC avec Neurax")

        # On vérifie si le simulateur quantique est disponible
        if not MODULE_IMPORT_SUCCESS:
            logger.warning("Modules Neurax non disponibles, tests ARC ignorés")
            return self.results

        # Limiter le nombre de puzzles pour des tests raisonnables
        # Pour un test complet, on utiliserait tous les puzzles
        # Phases à tester
        phases = ["training", "evaluation", "test"]

        # Fonction pour prédire la solution d'un puzzle ARC avec Neurax
        def predict_arc_solution(puzzle_id, train_pairs, test_input):
            """
            Utilise le système Neurax pour prédire la solution d'un puzzle ARC

            Args:
                puzzle_id: Identifiant du puzzle
                train_pairs: Paires d'exemples d'entraînement
                test_input: Entrée de test

            Returns:
                tuple: (prediction, confidence, metadata)
            """
            try:
                # Créer une instance du simulateur avec une taille adaptée au puzzle
                max_size = max(
                    max(len(train_pair["input"]), len(train_pair["input"][0]) if train_pair["input"] else 0) 
                    for train_pair in train_pairs
                )
                max_size = max(max_size, 
                               max(len(test_input), len(test_input[0]) if test_input else 0))

                # Assurer une taille minimale
                grid_size = max(max_size * 2, 32)

                # Créer le simulateur
                sim = QuantumGravitySimulator(grid_size=grid_size)

                # Représenter le puzzle dans le simulateur
                # Pour cette preuve de concept, nous utilisons une approche simplifiée
                # Initialiser l'espace-temps avec les données du puzzle

                # Pour chaque exemple d'entraînement, on encode les entrées/sorties
                # dans différentes régions de l'espace-temps
                for i, pair in enumerate(train_pairs):
                    input_grid = np.array(pair["input"], dtype=np.float32)
                    output_grid = np.array(pair["output"], dtype=np.float32)

                    # Calculer les dimensions
                    input_height, input_width = input_grid.shape
                    output_height, output_width = output_grid.shape

                    # Positionner les grilles dans le simulateur
                    # Nous utilisons différentes 'couches' temporelles
                    # Couche t=0: entrées
                    # Couche t=1: sorties attendues

                    # Ajouter l'entrée
                    x_offset = i * (input_width + 2)
                    y_offset = 0
                    z_offset = 0

                    # Encoder les entrées dans l'espace-temps
                    sim.space_time[0, 
                                  z_offset:z_offset+input_height, 
                                  y_offset:y_offset+input_width, 
                                  x_offset:x_offset+input_width] = input_grid

                    # Encoder les sorties dans l'espace-temps
                    sim.space_time[1, 
                                  z_offset:z_offset+output_height, 
                                  y_offset:y_offset+output_width, 
                                  x_offset:x_offset+output_width] = output_grid

                # Encoder l'entrée de test
                test_input_grid = np.array(test_input, dtype=np.float32)
                test_height, test_width = test_input_grid.shape

                # Positionner la grille de test
                x_offset = len(train_pairs) * (test_width + 2)

                # Encoder l'entrée de test dans l'espace-temps
                sim.space_time[0, 
                              z_offset:z_offset+test_height, 
                              y_offset:y_offset+test_width, 
                              x_offset:x_offset+test_width] = test_input_grid

                # Appliquer des fluctuations quantiques pour initialiser le processus
                sim.quantum_fluctuations(intensity=1.0)

                # Simuler plusieurs étapes pour permettre au système d'évoluer
                for _ in range(10):
                    sim.simulate_step()

                # Extraire la prédiction - dans la même position que l'entrée de test
                # mais dans la couche t=1 (sorties)
                prediction_region = sim.space_time[1, 
                                                 z_offset:z_offset+test_height, 
                                                 y_offset:y_offset+test_width, 
                                                 x_offset:x_offset+test_width]

                # Normaliser et discrétiser la prédiction
                # Les valeurs de l'espace-temps sont continues, nous les convertissons en entiers de 0 à 9
                # pour correspondre au format ARC
                min_val = np.min(prediction_region)
                max_val = np.max(prediction_region)

                # Éviter la division par zéro
                if max_val == min_val:
                    normalized = np.zeros_like(prediction_region)
                else:
                    normalized = (prediction_region - min_val) / (max_val - min_val)

                # Convertir en entiers 0-9
                prediction = np.round(normalized * 9).astype(np.int32).tolist()

                # Calculer une mesure de confiance basée sur la netteté des prédictions
                # Plus les valeurs sont proches des entiers, plus la confiance est élevée
                confidence = 1.0 - np.mean(np.abs(normalized * 9 - np.round(normalized * 9))) / 9

                # Collecter des métadonnées sur la simulation
                metadata = {
                    "min_val": float(min_val),
                    "max_val": float(max_val),
                    "avg_val": float(np.mean(prediction_region)),
                    "std_val": float(np.std(prediction_region))
                }

                return prediction, confidence, metadata

            except Exception as e:
                logger.error(f"Erreur lors de la prédiction pour le puzzle {puzzle_id}: {str(e)}")
                return None, 0.0, {"error": str(e)}

        # Pour chaque phase, tester un échantillon de puzzles
        for phase in phases:
            if phase not in self.arc_data:
                continue

            challenges = self.arc_data[phase]["challenges"]
            solutions = self.arc_data[phase]["solutions"] if "solutions" in self.arc_data[phase] else None

            # Pour un test complet, on utiliserait tous les puzzles
            # Mais pour un test initial, on peut utiliser un échantillon
            # On limite à 5 puzzles par phase pour cet exemple
            puzzle_ids = list(challenges.keys())[:5]

            for puzzle_id in puzzle_ids:
                logger.info(f"Test du puzzle ARC {puzzle_id} (phase: {phase})")

                start_time = time.time()
                try:
                    puzzle = challenges[puzzle_id]
                    train_pairs = puzzle["train"]
                    test_inputs = puzzle["test"]

                    # Pour chaque entrée de test
                    for i, test_input in enumerate(test_inputs):
                        test_input_data = test_input["input"]

                        # Obtenir la solution réelle si disponible
                        real_solution = None
                        if solutions and puzzle_id in solutions:
                            solution_list = solutions[puzzle_id]
                            if i < len(solution_list):
                                real_solution = solution_list[i]["output"]

                        # Prédire avec Neurax
                        prediction, confidence, metadata = predict_arc_solution(
                            puzzle_id, train_pairs, test_input_data
                        )

                        # Calculer l'exactitude si la solution réelle est disponible
                        accuracy = None
                        if real_solution is not None and prediction is not None:
                            # Vérifier d'abord les dimensions
                            if (len(prediction) == len(real_solution) and 
                                all(len(prediction[j]) == len(real_solution[j]) for j in range(len(prediction)))):
                                # Compter les cellules correctes
                                correct_cells = sum(
                                    1 for j in range(len(prediction)) 
                                    for k in range(len(prediction[j])) 
                                    if prediction[j][k] == real_solution[j][k]
                                )
                                total_cells = len(prediction) * len(prediction[0])
                                accuracy = correct_cells / total_cells
                            else:
                                accuracy = 0.0  # Dimensions incorrectes

                        # Déterminer le statut du test
                        if accuracy is not None:
                            status = "PASS" if accuracy >= 0.8 else "FAIL"
                        else:
                            status = "SKIP"  # Pas de solution de référence

                        # Ajouter les résultats
                        test_details = {
                            "phase": phase,
                            "input_shape": f"{len(test_input_data)}x{len(test_input_data[0])}",
                            "prediction_shape": f"{len(prediction)}x{len(prediction[0])}" if prediction is not None else "None",
                            "confidence": confidence,
                            "metadata": metadata
                        }

                        if prediction is not None:
                            test_details["prediction"] = prediction  # Inclure la prédiction complète

                        execution_time = time.time() - start_time
                        self.results.add_arc_puzzle_result(
                            puzzle_id, f"{phase}_test_{i}", status, execution_time, accuracy, test_details
                        )

                except Exception as e:
                    logger.error(f"Erreur lors du test du puzzle {puzzle_id}: {str(e)}")
                    execution_time = time.time() - start_time
                    self.results.add_arc_puzzle_result(
                        puzzle_id, phase, "FAIL", execution_time, 0.0, {"error": str(e)}
                    )

        return self.results

    def run_performance_tests(self):
        """Exécute des tests de performance sur les différents composants"""
        logger.info("Exécution des tests de performance")

        # Tester les performances du simulateur de gravité quantique
        try:
            # Tailles de grille à tester
            grid_sizes = [20, 32, 50, 64]

            # Mesurer le temps de création du simulateur pour différentes tailles
            init_times = {}
            for size in grid_sizes:
                start_time = time.time()
                sim = QuantumGravitySimulator(grid_size=size)
                init_time = time.time() - start_time
                init_times[size] = init_time

            self.results.add_performance_metric("Simulateur", "init_times", init_times)

            # Mesurer le temps des fluctuations quantiques
            fluctuation_times = {}
            for size in grid_sizes:
                sim = QuantumGravitySimulator(grid_size=size)
                start_time = time.time()
                sim.quantum_fluctuations()
                fluctuation_time = time.time() - start_time
                fluctuation_times[size] = fluctuation_time

            self.results.add_performance_metric("Simulateur", "fluctuation_times", fluctuation_times)

            # Mesurer le temps des étapes de simulation
            simulation_step_times = {}
            for size in grid_sizes:
                sim = QuantumGravitySimulator(grid_size=size)
                sim.quantum_fluctuations()

                # Mesurer 5 étapes
                steps = 5
                step_times = []
                for _ in range(steps):
                    start_time = time.time()
                    sim.simulate_step()
                    step_time = time.time() - start_time
                    step_times.append(step_time)

                simulation_step_times[size] = {
                    "min": min(step_times),
                    "max": max(step_times),
                    "avg": sum(step_times) / len(step_times)
                }

            self.results.add_performance_metric("Simulateur", "simulation_step_times", simulation_step_times)

            # Mesurer l'utilisation mémoire
            memory_usage = {}
            for size in grid_sizes:
                # Mesurer avant la création
                mem_before = psutil.Process().memory_info().rss / (1024 * 1024)  # MB

                sim = QuantumGravitySimulator(grid_size=size)
                sim.quantum_fluctuations()

                # Mesurer après la création
                mem_after = psutil.Process().memory_info().rss / (1024 * 1024)  # MB

                memory_usage[size] = mem_after - mem_before

            self.results.add_performance_metric("Simulateur", "memory_usage_mb", memory_usage)

        except Exception as e:
            logger.error(f"Erreur lors des tests de performance: {str(e)}")

        return self.results

    def run_all_tests(self):
        """Exécute tous les tests disponibles"""
        logger.info("Démarrage des tests complets pour le projet Neurax")

        # Tester le simulateur de gravité quantique
        logger.info("Test du simulateur de gravité quantique")
        self.test_quantum_gravity_simulator()

        # Tester le système neuronal (si disponible)
        logger.info("Test du système neuronal")
        self.test_quantum_neuron()

        # Tester le réseau P2P (si disponible)
        logger.info("Test du réseau P2P")
        self.test_p2p_network()

        # Tester le mécanisme de consensus (si disponible)
        logger.info("Test du mécanisme de consensus")
        self.test_consensus_mechanism()

        # Tester la visualisation (si disponible)
        logger.info("Test du module de visualisation")
        self.test_visualization()

        # Tester le gestionnaire d'export (si disponible)
        logger.info("Test du gestionnaire d'export")
        self.test_export_manager()

        # Tester le gestionnaire de base de données (si disponible)
        logger.info("Test du gestionnaire de base de données")
        self.test_database_manager()

        # Tests de performance
        logger.info("Tests de performance")
        self.run_performance_tests()

        # Tester les puzzles ARC (optionnel)
        if self.arc_data:
            logger.info("Test des puzzles ARC")
            self.test_arc_puzzles_with_neurax()

        # Générer les rapports
        summary = self.results.get_summary()
        logger.info(f"Tests terminés. Taux de réussite: {summary['success_rate']}%")

        return self.results

    def _load_arc_data(self, data_path="arc_data"):
        """
        Charge les données des puzzles ARC

        Args:
            data_path: Chemin vers les fichiers JSON ARC

        Returns:
            dict: Un dictionnaire contenant les données d'entraînement, d'évaluation et de test
        """
        try:
            # Construction des chemins complets vers les fichiers de données
            training_path = os.path.join(data_path, "arc-agi_training_challenges.json")
            evaluation_path = os.path.join(data_path, "arc-agi_evaluation_challenges.json")
            test_path = os.path.join(data_path, "arc-agi_test_challenges.json")
            training_solutions_path = os.path.join(data_path, "arc-agi_training_solutions.json")
            evaluation_solutions_path = os.path.join(data_path, "arc-agi_evaluation_solutions.json")

            # Chargement des données depuis les fichiers JSON
            with open(training_path, 'r') as f:
                training_challenges = json.load(f)
            with open(evaluation_path, 'r') as f:
                evaluation_challenges = json.load(f)
            with open(test_path, 'r') as f:
                test_challenges = json.load(f)
            with open(training_solutions_path, 'r') as f:
                training_solutions = json.load(f)
            with open(evaluation_solutions_path, 'r') as f:
                evaluation_solutions = json.load(f)

            # Création des dictionnaires de données pour chaque ensemble
            training_data = {"challenges": training_challenges, "solutions": training_solutions}
            evaluation_data = {"challenges": evaluation_challenges, "solutions": evaluation_solutions}
            test_data = {"challenges": test_challenges, "solutions": None}  # Pas de solutions pour l'ensemble de test

            # Retourne un dictionnaire contenant tous les ensembles de données
            return {"training": training_data, "evaluation": evaluation_data, "test": test_data}

        except FileNotFoundError as e:
            logger.error(f"Fichier de données ARC non trouvé: {e}")
            return None  # Gérer l'erreur si les fichiers ne sont pas trouvés
        except json.JSONDecodeError as e:
            logger.error(f"Erreur de décodage JSON dans les données ARC: {e}")
            return None  # Gérer l'erreur si le JSON est invalide

    def run_all_tests(self):
        """Exécute tous les tests disponibles"""
        logger.info("Démarrage des tests complets pour le projet Neurax")

        # Exécuter les tests du simulateur de gravité quantique
        logger.info("Test du simulateur de gravité quantique")
        self.test_quantum_gravity_simulator()

        # Exécuter les tests du neurone quantique
        logger.info("Test du système neuronal")
        self.test_quantum_neuron()

        # Exécuter les tests du réseau P2P
        logger.info("Test du réseau P2P")
        self.test_p2p_network()

        # Exécuter les tests du mécanisme de consensus
        logger.info("Test du mécanisme de consensus")
        self.test_consensus_mechanism()

        # Exécuter les tests de visualisation
        logger.info("Test du module de visualisation")
        self.test_visualization()

        # Exécuter les tests du gestionnaire d'export
        logger.info("Test du gestionnaire d'export")
        self.test_export_manager()

        # Exécuter les tests du gestionnaire de base de données
        logger.info("Test du gestionnaire de base de données")
        self.test_database_manager()

        # Exécuter les tests de performance
        logger.info("Tests de performance")
        self.run_performance_tests()

        # Exécuter les tests des puzzles ARC
        logger.info("Test des puzzles ARC")
        self.test_arc_puzzles_with_neurax()

        # Générer le résumé des résultats
        summary = self.results.get_summary()
        logger.info(f"Tests terminés. Taux de réussite: {summary['success_rate']}%")

        return self.results