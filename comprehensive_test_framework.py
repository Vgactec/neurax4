#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import csv
import logging
import numpy as np
import traceback
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
from functools import lru_cache

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

# Nombre de processus à utiliser pour le traitement parallèle
NUM_PROCESSES = max(1, multiprocessing.cpu_count() - 1)
logger.info(f"Initialisation avec {NUM_PROCESSES} processus parallèles")

# Importation du simulateur de gravité quantique
try:
    from quantum_gravity_sim import QuantumGravitySimulator
    logger.info("Simulateur de gravité quantique importé depuis le module racine")
    SIMULATOR_SOURCE = "root"
except ImportError:
    try:
        from core.quantum_sim.simulator import QuantumGravitySimulator
        logger.info("Simulateur de gravité quantique importé depuis le module core")
        SIMULATOR_SOURCE = "core"
    except ImportError:
        logger.error("Impossible d'importer le simulateur de gravité quantique")
        SIMULATOR_SOURCE = None


class TestResults:
    """
    Classe pour gérer et analyser les résultats des tests
    """
    
    def __init__(self):
        """Initialisation des résultats de test"""
        self.timestamp = datetime.now()
        self.results = []
        self.puzzle_results = {}  # Résultats détaillés par puzzle
        
    def setup_method(self):
        """Réinitialise les résultats"""
        self.timestamp = datetime.now()
        self.results = []
        self.puzzle_results = {}

    def add_result(self, component: str, status: str, duration: float, details: Dict[str, Any]):
        """
        Ajoute un résultat de test
        
        Args:
            component: Nom du composant testé
            status: Statut du test (PASS/FAIL)
            duration: Durée du test en secondes
            details: Détails des résultats
        """
        self.results.append({
            "component": component,
            "status": status,
            "duration": duration,
            "details": details
        })
        
    def add_puzzle_result(self, puzzle_id: str, phase: str, result: Dict[str, Any]):
        """
        Ajoute un résultat spécifique à un puzzle
        
        Args:
            puzzle_id: Identifiant unique du puzzle
            phase: Phase de test (training/evaluation/test)
            result: Résultats détaillés
        """
        if puzzle_id not in self.puzzle_results:
            self.puzzle_results[puzzle_id] = {}
        
        self.puzzle_results[puzzle_id][phase] = result

    def get_summary(self) -> Dict[str, Any]:
        """
        Génère un résumé des résultats des tests
        
        Returns:
            Dictionnaire contenant le résumé des résultats
        """
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r["status"] == "PASS")
        failed_tests = total_tests - passed_tests
        
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        # Calculer la précision moyenne des composants
        average_accuracy = 0.0
        accuracy_count = 0
        
        for result in self.results:
            if "details" in result and "average_accuracy" in result["details"]:
                average_accuracy += result["details"]["average_accuracy"]
                accuracy_count += 1
        
        if accuracy_count > 0:
            average_accuracy /= accuracy_count
            
        # Compter les puzzles
        total_puzzles = len(self.puzzle_results)
        
        # Statistiques par phase
        phases = {"training": 0, "evaluation": 0, "test": 0}
        success_by_phase = {"training": 0, "evaluation": 0, "test": 0}
        
        for puzzle_id, phases_data in self.puzzle_results.items():
            for phase, data in phases_data.items():
                if phase in phases:
                    phases[phase] += 1
                    if data.get("converged", False):
                        success_by_phase[phase] += 1
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": success_rate,
            "average_accuracy": average_accuracy,
            "puzzle_count": total_puzzles,
            "puzzles_by_phase": phases,
            "success_by_phase": success_by_phase
        }
    
    def get_arc_results(self) -> Dict[str, Dict[str, Any]]:
        """
        Retourne les résultats spécifiques aux puzzles ARC
        
        Returns:
            Dictionnaire des résultats par phase et global
        """
        arc_results = {}
        
        # Résultats globaux
        for result in self.results:
            if result["component"] == "Apprentissage Complet" and "details" in result:
                details = result["details"]
                if "successful_puzzles" in details:
                    arc_results["global"] = {
                        "status": result["status"],
                        "total": details.get("total_puzzles", 0),
                        "success": details.get("successful_puzzles", 0),
                        "accuracy": details.get("average_accuracy", 0),
                        "phase": "all"
                    }
        
        # Résultats par puzzle
        for puzzle_id, phases_data in self.puzzle_results.items():
            for phase, data in phases_data.items():
                phase_key = f"puzzle_{puzzle_id}_{phase}"
                arc_results[phase_key] = {
                    "puzzle_id": puzzle_id,
                    "status": "PASS" if data.get("converged", False) else "FAIL",
                    "accuracy": data.get("final_accuracy", 0),
                    "epochs": data.get("epochs", 0),
                    "phase": phase
                }
                
        # Résultats par phase
        phases = ["training", "evaluation", "test"]
        for phase in phases:
            puzzles_in_phase = [
                data for puzzle_id, phases_data in self.puzzle_results.items()
                for p, data in phases_data.items() if p == phase
            ]
            
            if puzzles_in_phase:
                total = len(puzzles_in_phase)
                success = sum(1 for p in puzzles_in_phase if p.get("converged", False))
                avg_accuracy = sum(p.get("final_accuracy", 0) for p in puzzles_in_phase) / total
                
                arc_results[f"phase_{phase}"] = {
                    "status": "PASS" if success > 0 else "FAIL",
                    "total": total,
                    "success": success,
                    "accuracy": avg_accuracy,
                    "phase": phase
                }
                
        return arc_results

    def export_to_json(self, filename: str = "arc_tests_results.json"):
        """
        Exporte les résultats au format JSON
        
        Args:
            filename: Nom du fichier de sortie
        """
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)
            
    def export_puzzle_results_to_json(self, filename: str = "arc_puzzles_detailed_results.json"):
        """
        Exporte les résultats détaillés des puzzles au format JSON
        
        Args:
            filename: Nom du fichier de sortie
        """
        with open(filename, "w") as f:
            json.dump(self.puzzle_results, f, indent=2)

    def export_to_csv(self, filename: str = "arc_tests_results.csv"):
        """
        Exporte les résultats au format CSV
        
        Args:
            filename: Nom du fichier de sortie
        """
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
    
    def export_arc_results_to_csv(self, filename: str = "arc_puzzles_results.csv"):
        """
        Exporte les résultats des puzzles ARC au format CSV
        
        Args:
            filename: Nom du fichier de sortie
        """
        with open(filename, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Puzzle_ID", "Status", "Accuracy", "Epochs", "Phase"])
            
            # Résultats globaux
            arc_results = self.get_arc_results()
            if "global" in arc_results:
                writer.writerow([
                    "GLOBAL",
                    arc_results["global"]["status"],
                    arc_results["global"]["accuracy"],
                    "",
                    "ALL"
                ])
            
            # Résultats par phase
            for phase in ["training", "evaluation", "test"]:
                phase_key = f"phase_{phase}"
                if phase_key in arc_results:
                    writer.writerow([
                        f"PHASE_{phase.upper()}",
                        arc_results[phase_key]["status"],
                        arc_results[phase_key]["accuracy"],
                        "",
                        phase
                    ])
            
            # Résultats détaillés par puzzle
            for key, result in arc_results.items():
                if key.startswith("puzzle_"):
                    writer.writerow([
                        result["puzzle_id"],
                        result["status"],
                        result["accuracy"],
                        result["epochs"],
                        result["phase"]
                    ])
    
    def generate_detailed_report(self, filename: str = "analyse_resultats_reels.md"):
        """
        Génère un rapport détaillé des résultats au format Markdown
        
        Args:
            filename: Nom du fichier de sortie
            
        Returns:
            Contenu du rapport généré
        """
        report = f"""# Rapport d'Analyse Détaillée des Tests Neurax2

## Vue d'ensemble

Date des tests: {self.timestamp.strftime('%d-%m-%Y %H:%M:%S')}

## Résultats globaux

"""
        summary = self.get_summary()
        report += f"""- **Nombre total de tests:** {summary.get('total_tests', 0)}
- **Tests réussis:** {summary.get('passed_tests', 0)}
- **Tests échoués:** {summary.get('failed_tests', 0)}
- **Taux de réussite global:** {summary.get('success_rate', 0)*100:.2f}%
- **Précision moyenne:** {summary.get('average_accuracy', 0)*100:.2f}%
- **Nombre total de puzzles ARC traités:** {summary.get('puzzle_count', 0)}

## Résultats par phase

- **Phase d'entraînement:** {summary.get('puzzles_by_phase', {}).get('training', 0)} puzzles, {summary.get('success_by_phase', {}).get('training', 0)} réussis
- **Phase d'évaluation:** {summary.get('puzzles_by_phase', {}).get('evaluation', 0)} puzzles, {summary.get('success_by_phase', {}).get('evaluation', 0)} réussis
- **Phase de test:** {summary.get('puzzles_by_phase', {}).get('test', 0)} puzzles, {summary.get('success_by_phase', {}).get('test', 0)} réussis

## Détails des composants testés

"""
        # Ajouter les détails de chaque test
        for i, result in enumerate(self.results):
            report += f"""### Test {i+1}: {result['component']}

- **Statut:** {result['status']}
- **Durée:** {result['duration']:.4f} secondes
"""
            if 'details' in result and result['details']:
                report += "- **Détails:**\n"
                for key, value in result['details'].items():
                    if isinstance(value, (list, dict)):
                        report += f"  - {key}: {json.dumps(value, indent=2)}\n"
                    else:
                        report += f"  - {key}: {value}\n"
            
            report += "\n"
        
        # Ajouter des informations sur les puzzles
        arc_results = self.get_arc_results()
        phase_keys = [k for k in arc_results.keys() if k.startswith("phase_")]
        if phase_keys:
            report += "## Statistiques par phase ARC\n\n"
            
            for phase_key in phase_keys:
                phase_data = arc_results[phase_key]
                phase_name = phase_data["phase"].capitalize()
                success_rate = phase_data["success"] / phase_data["total"] * 100 if phase_data["total"] > 0 else 0
                
                report += f"""### Phase {phase_name}

- **Nombre de puzzles:** {phase_data['total']}
- **Puzzles résolus:** {phase_data['success']} ({success_rate:.2f}%)
- **Précision moyenne:** {phase_data['accuracy']*100:.2f}%

"""
        
        # Ajouter des détails sur les puzzles les plus difficiles/faciles
        puzzle_results = [
            v for k, v in arc_results.items() 
            if k.startswith("puzzle_") and "accuracy" in v and "epochs" in v
        ]
        
        if puzzle_results:
            # Trier par précision (décroissante)
            best_puzzles = sorted(puzzle_results, key=lambda r: -r["accuracy"])[:5]
            worst_puzzles = sorted(puzzle_results, key=lambda r: r["accuracy"])[:5]
            
            report += "## Puzzles avec les meilleurs résultats\n\n"
            for p in best_puzzles:
                report += f"- **{p['puzzle_id']}** (Phase: {p['phase']}): Précision {p['accuracy']*100:.2f}%, {p['epochs']} epochs\n"
            
            report += "\n## Puzzles avec les résultats les plus faibles\n\n"
            for p in worst_puzzles:
                report += f"- **{p['puzzle_id']}** (Phase: {p['phase']}): Précision {p['accuracy']*100:.2f}%, {p['epochs']} epochs\n"
        
        # Conclusion
        report += """
## Conclusion

Cette analyse représente l'état actuel du système Neurax2 en termes de performances et de fonctionnalités sur les puzzles de la compétition ARC-Prize-2025. Les résultats mettent en évidence les forces et les faiblesses du système actuel, et peuvent guider les améliorations futures.
"""
        
        # Écrire le rapport dans un fichier
        with open(filename, "w") as f:
            f.write(report)
        
        return report


class TestSuite:
    """
    Suite de tests complète pour le système Neurax
    """
    
    def __init__(self, arc_data_path: str = "./neurax_complet/arc_data"):
        """
        Initialise la suite de tests
        
        Args:
            arc_data_path: Chemin vers les données ARC
        """
        self.arc_data_path = arc_data_path
        self.results = TestResults()
        self.min_epochs = 10
        self.max_epochs = 1000
        self.convergence_threshold = 0.99
        
        # Configuration du logger
        self.logger = logging.getLogger("ARC_Tests")
        fh = logging.FileHandler('arc_complete_tests.log')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - Puzzle %(puzzle_id)s: %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def load_arc_data(self, phase: str = "training") -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Charge les données ARC pour une phase spécifique
        
        Args:
            phase: Phase à charger (training/evaluation/test)
            
        Returns:
            Tuple contenant (puzzles, solutions)
        """
        if phase not in ["training", "evaluation", "test"]:
            raise ValueError(f"Phase invalide: {phase}. Doit être 'training', 'evaluation' ou 'test'")
        
        # Chemins des fichiers
        challenges_path = os.path.join(self.arc_data_path, f"arc-agi_{phase}_challenges.json")
        
        # Les solutions ne sont disponibles que pour les phases training et evaluation
        solutions_path = os.path.join(self.arc_data_path, f"arc-agi_{phase}_solutions.json") if phase != "test" else None
        
        self.logger.info(f"Chargement des puzzles de la phase {phase}", extra={'puzzle_id': 'INIT'})
        
        try:
            # Chargement des puzzles
            with open(challenges_path, 'r') as f:
                puzzles = json.load(f)
                
            # Chargement des solutions (si disponibles)
            solutions = {}
            if solutions_path and phase != "test":
                with open(solutions_path, 'r') as f:
                    solutions = json.load(f)
            
            self.logger.info(f"Chargement réussi: {len(puzzles)} puzzles de {phase}", extra={'puzzle_id': 'INIT'})
            return puzzles, solutions
            
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement des puzzles {phase}: {str(e)}", extra={'puzzle_id': 'INIT'})
            return {}, {}

    def process_puzzle(self, puzzle_id: str, puzzle_data: Dict[str, Any], 
                      solutions: Optional[Dict[str, Any]] = None, 
                      phase: str = "training") -> Dict[str, Any]:
        """
        Traite un puzzle ARC
        
        Args:
            puzzle_id: Identifiant du puzzle
            puzzle_data: Données du puzzle
            solutions: Solutions (optionnel pour la phase de test)
            phase: Phase de traitement
            
        Returns:
            Résultat du traitement
        """
        start_time = time.time()
        current_accuracy = 0.0
        epochs = 0
        converged = False
        
        try:
            # Boucle d'entraînement
            while epochs < self.max_epochs and current_accuracy < self.convergence_threshold:
                # Traiter tous les exemples d'entraînement
                example_accuracies = []
                
                for example_idx, train_pair in enumerate(puzzle_data["train"]):
                    try:
                        # Initialiser le simulateur avec une taille adaptée au puzzle
                        input_shape = np.array(train_pair["input"]).shape
                        output_shape = np.array(train_pair["output"]).shape
                        grid_size = max(max(input_shape), max(output_shape), 32)
                        
                        sim = QuantumGravitySimulator(grid_size=grid_size, time_steps=8)
                        
                        # Convertir les grilles en tableaux NumPy
                        input_grid = np.array(train_pair["input"])
                        output_grid = np.array(train_pair["output"])
                        
                        # Simulation et apprentissage
                        sim.quantum_fluctuations(intensity=1.5)
                        sim.simulate_step()
                        
                        # Évaluer la précision
                        prediction = sim.space_time[-1][:output_shape[0], :output_shape[1]]
                        example_accuracy = np.mean(np.isclose(prediction, output_grid))
                        example_accuracies.append(example_accuracy)
                        
                    except Exception as e:
                        self.logger.error(f"Erreur sur l'exemple {example_idx} du puzzle {puzzle_id}: {str(e)}", 
                                      extra={'puzzle_id': puzzle_id})
                
                # Calculer la précision moyenne sur tous les exemples
                if example_accuracies:
                    current_accuracy = sum(example_accuracies) / len(example_accuracies)
                
                # Vérifier la convergence
                converged = current_accuracy >= self.convergence_threshold
                
                # Incrémenter le nombre d'epochs
                epochs += 1
                
                # Si convergence atteinte, sortir de la boucle
                if converged:
                    self.logger.info(f"Puzzle {puzzle_id} résolu en {epochs} epochs avec précision {current_accuracy:.4f}", 
                                  extra={'puzzle_id': puzzle_id})
                    break
            
            # Résultats finaux
            duration = time.time() - start_time
            
            result = {
                "puzzle_id": puzzle_id,
                "phase": phase,
                "epochs": epochs,
                "final_accuracy": current_accuracy,
                "converged": converged,
                "duration": duration
            }
            
            # Ajouter les résultats de test si disponibles
            if phase == "training" and "test" in puzzle_data and len(puzzle_data["test"]) > 0:
                test_accuracies = []
                
                for test_idx, test_pair in enumerate(puzzle_data["test"]):
                    try:
                        # Initialiser le simulateur
                        sim = QuantumGravitySimulator(grid_size=grid_size, time_steps=8)
                        
                        # Convertir la grille d'entrée
                        test_input = np.array(test_pair["input"])
                        
                        # Appliquer la simulation
                        sim.quantum_fluctuations(intensity=1.5)
                        sim.simulate_step()
                        
                        # Obtenir la prédiction
                        prediction = sim.space_time[-1][:test_input.shape[0], :test_input.shape[1]]
                        
                        # Vérifier avec la solution attendue si disponible
                        if solutions and puzzle_id in solutions and "test" in solutions[puzzle_id]:
                            expected_output = np.array(solutions[puzzle_id]["test"][test_idx]["output"])
                            test_accuracy = np.mean(np.isclose(prediction, expected_output))
                            test_accuracies.append(test_accuracy)
                    
                    except Exception as e:
                        self.logger.error(f"Erreur sur le test {test_idx} du puzzle {puzzle_id}: {str(e)}", 
                                       extra={'puzzle_id': puzzle_id})
                
                # Ajouter la précision moyenne des tests
                if test_accuracies:
                    result["test_accuracy"] = sum(test_accuracies) / len(test_accuracies)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur générale sur le puzzle {puzzle_id}: {str(e)}", 
                           extra={'puzzle_id': puzzle_id})
            return {
                "puzzle_id": puzzle_id,
                "phase": phase,
                "epochs": epochs,
                "final_accuracy": current_accuracy,
                "converged": False,
                "error": str(e),
                "duration": time.time() - start_time
            }

    def process_puzzles(self, phase: str = "training", max_puzzles: int = 1000) -> Dict[str, Any]:
        """
        Traite tous les puzzles d'une phase
        
        Args:
            phase: Phase à traiter
            max_puzzles: Nombre maximum de puzzles à traiter
            
        Returns:
            Résultats détaillés du traitement
        """
        start_time = time.time()
        
        try:
            # Charger les puzzles
            puzzles, solutions = self.load_arc_data(phase)
            
            if not puzzles:
                raise ValueError(f"Aucun puzzle chargé pour la phase {phase}")
            
            # Limiter le nombre de puzzles si nécessaire
            puzzle_ids = list(puzzles.keys())
            if max_puzzles < len(puzzle_ids):
                puzzle_ids = puzzle_ids[:max_puzzles]
                self.logger.info(f"Limitation à {max_puzzles} puzzles pour la phase {phase}", 
                               extra={'puzzle_id': 'LIMIT'})
            
            # Résultats
            processed_puzzles = []
            successful_puzzles = []
            failed_puzzles = []
            
            # Traiter chaque puzzle
            total_puzzles = len(puzzle_ids)
            for idx, puzzle_id in enumerate(puzzle_ids):
                # Afficher la progression
                progress = (idx + 1) / total_puzzles * 100
                self.logger.info(f"Traitement puzzle {idx+1}/{total_puzzles} ({progress:.1f}%) - ID: {puzzle_id} - Phase: {phase}", 
                               extra={'puzzle_id': puzzle_id})
                
                # Vérifier que le puzzle existe
                if puzzle_id not in puzzles:
                    self.logger.warning(f"Puzzle ID {puzzle_id} non trouvé dans les données de {phase}", 
                                     extra={'puzzle_id': puzzle_id})
                    continue
                
                # Traiter le puzzle
                result = self.process_puzzle(puzzle_id, puzzles[puzzle_id], solutions, phase)
                processed_puzzles.append(result)
                
                # Ajouter le résultat dans la structure de données principale
                self.results.add_puzzle_result(puzzle_id, phase, result)
                
                # Classifier le résultat
                if result.get("converged", False):
                    successful_puzzles.append(result)
                else:
                    failed_puzzles.append(result)
            
            # Analyser les résultats
            if processed_puzzles:
                avg_epochs = np.mean([r.get("epochs", 0) for r in processed_puzzles])
                avg_accuracy = np.mean([r.get("final_accuracy", 0) for r in processed_puzzles])
                min_epochs = min([r.get("epochs", 0) for r in processed_puzzles]) if processed_puzzles else 0
                max_epochs = max([r.get("epochs", 0) for r in processed_puzzles]) if processed_puzzles else 0
                avg_duration = np.mean([r.get("duration", 0) for r in processed_puzzles])
            else:
                avg_epochs = avg_accuracy = min_epochs = max_epochs = avg_duration = 0
            
            # Statistiques
            details = {
                "phase": phase,
                "total_puzzles": total_puzzles,
                "processed_puzzles": len(processed_puzzles),
                "successful_puzzles": len(successful_puzzles),
                "failed_puzzles": len(failed_puzzles),
                "success_rate": len(successful_puzzles) / total_puzzles if total_puzzles > 0 else 0,
                "average_accuracy": avg_accuracy,
                "average_epochs": avg_epochs,
                "min_epochs": min_epochs,
                "max_epochs": max_epochs,
                "average_duration": avg_duration,
                "total_duration": time.time() - start_time
            }
            
            # Ajouter les exemples de puzzles faciles/difficiles
            if successful_puzzles:
                easiest_puzzles = sorted(successful_puzzles, key=lambda r: r.get("epochs", float('inf')))[:5]
                hardest_puzzles = sorted(successful_puzzles, key=lambda r: -r.get("epochs", 0))[:5]
                
                details["easiest_puzzles"] = [
                    {"id": p["puzzle_id"], "epochs": p.get("epochs", 0), "accuracy": p.get("final_accuracy", 0)} 
                    for p in easiest_puzzles
                ]
                details["hardest_puzzles"] = [
                    {"id": p["puzzle_id"], "epochs": p.get("epochs", 0), "accuracy": p.get("final_accuracy", 0)} 
                    for p in hardest_puzzles
                ]
            
            # Ajouter le résultat à la liste des résultats
            component_name = f"Apprentissage {phase.capitalize()}"
            self.results.add_result(
                component_name,
                "PASS" if len(successful_puzzles) > 0 else "FAIL",
                time.time() - start_time,
                details
            )
            
            return details
            
        except Exception as e:
            self.logger.error(f"Erreur lors du traitement de la phase {phase}: {str(e)}")
            
            # Ajouter un résultat d'échec
            component_name = f"Apprentissage {phase.capitalize()}"
            self.results.add_result(
                component_name,
                "FAIL",
                time.time() - start_time,
                {"error": str(e), "phase": phase}
            )
            
            return {"error": str(e), "phase": phase}

    def test_quantum_gravity_simulator(self) -> Dict[str, Any]:
        """
        Teste le simulateur de gravité quantique
        
        Returns:
            Résultats du test
        """
        start_time = time.time()
        
        try:
            # Tester l'initialisation
            sim = QuantumGravitySimulator(grid_size=32, time_steps=8)
            
            # Vérifier que l'espace-temps est correctement initialisé
            has_space_time = hasattr(sim, "space_time")
            space_time_shape = sim.space_time.shape if has_space_time else None
            
            # Tester les fluctuations quantiques
            if has_space_time:
                # Enregistrer l'état initial
                initial_state = sim.space_time.copy()
                
                # Appliquer des fluctuations
                sim.quantum_fluctuations(intensity=1.0)
                
                # Vérifier que l'état a changé
                state_changed = not np.array_equal(initial_state, sim.space_time)
                
                # Tester la simulation
                sim.simulate_step()
                
                # Vérifier que la simulation a modifié l'espace-temps
                simulation_worked = not np.array_equal(initial_state, sim.space_time)
            else:
                state_changed = simulation_worked = False
            
            # Résultats
            status = "PASS" if has_space_time and state_changed and simulation_worked else "FAIL"
            details = {
                "grid_size": 32,
                "time_steps": 8,
                "has_space_time": has_space_time,
                "space_time_shape": space_time_shape,
                "state_changed_after_fluctuations": state_changed,
                "simulation_worked": simulation_worked
            }
            
        except Exception as e:
            status = "FAIL"
            details = {"error": str(e)}
        
        # Enregistrer le résultat
        duration = time.time() - start_time
        self.results.add_result(
            "Simulateur Quantique",
            status,
            duration,
            details
        )
        
        return details

    def run_all_tests(self, training_puzzles: int = 1000, 
                    evaluation_puzzles: int = 120, 
                    test_puzzles: int = 240) -> TestResults:
        """
        Exécute tous les tests complets sur l'ensemble des puzzles
        
        Args:
            training_puzzles: Nombre de puzzles d'entraînement à traiter
            evaluation_puzzles: Nombre de puzzles d'évaluation à traiter
            test_puzzles: Nombre de puzzles de test à traiter
            
        Returns:
            Résultats des tests
        """
        start_time = time.time()
        logger.info("=== DÉMARRAGE DES TESTS COMPLETS NEURAX2 ===")
        
        # Tester le simulateur de gravité quantique
        logger.info("Test du simulateur de gravité quantique")
        simulator_results = self.test_quantum_gravity_simulator()
        
        if simulator_results.get("error"):
            logger.error(f"Erreur lors du test du simulateur: {simulator_results.get('error')}")
            return self.results
        
        # Traiter les puzzles d'entraînement
        if training_puzzles > 0:
            logger.info(f"Traitement de {training_puzzles} puzzles d'entraînement")
            self.process_puzzles("training", training_puzzles)
        
        # Traiter les puzzles d'évaluation
        if evaluation_puzzles > 0:
            logger.info(f"Traitement de {evaluation_puzzles} puzzles d'évaluation")
            self.process_puzzles("evaluation", evaluation_puzzles)
        
        # Traiter les puzzles de test
        if test_puzzles > 0:
            logger.info(f"Traitement de {test_puzzles} puzzles de test")
            self.process_puzzles("test", test_puzzles)
        
        # Exporter les résultats
        logger.info("Exportation des résultats")
        self.results.export_to_json()
        self.results.export_to_csv()
        self.results.export_arc_results_to_csv()
        self.results.export_puzzle_results_to_json()
        
        # Générer le rapport détaillé
        logger.info("Génération du rapport d'analyse")
        self.results.generate_detailed_report()
        
        # Résumé final
        total_duration = time.time() - start_time
        summary = self.results.get_summary()
        
        logger.info("=== RÉSULTATS TESTS NEURAX2 ===")
        logger.info(f"Durée totale: {total_duration:.2f} secondes")
        logger.info(f"Puzzles traités: {summary.get('puzzle_count', 0)}")
        logger.info(f"Taux de réussite: {summary.get('success_rate', 0)*100:.2f}%")
        logger.info(f"Précision moyenne: {summary.get('average_accuracy', 0)*100:.2f}%")
        logger.info("=== FIN DES TESTS ===")
        
        return self.results