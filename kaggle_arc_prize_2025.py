#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script d'intégration pour la compétition Kaggle ARC-Prize-2025
Adapté aux règles spécifiques de la compétition et optimisé pour l'environnement Kaggle
"""

import os
import sys
import json
import time
import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
import argparse

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("kaggle_arc.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("KaggleARC")

# Détection de l'environnement Kaggle
IN_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

# Drapeau pour indiquer si nous sommes en mode test ou en production
TEST_MODE = not IN_KAGGLE

class ArcPrizeAdapter:
    """
    Adaptateur pour la compétition ARC-Prize-2025 sur Kaggle
    """
    
    def __init__(self, 
                neurax_path: str = ".",
                training_path: str = "./training",
                evaluation_path: str = "./evaluation",
                test_path: str = "./test",
                output_dir: str = "./output",
                use_gpu: bool = True,
                optimize_for_mobile: bool = False):
        """
        Initialise l'adaptateur pour la compétition ARC-Prize-2025
        
        Args:
            neurax_path: Chemin vers les modules Neurax
            training_path: Chemin vers les données d'entraînement
            evaluation_path: Chemin vers les données d'évaluation
            test_path: Chemin vers les données de test
            output_dir: Répertoire de sortie pour les résultats
            use_gpu: Utiliser le GPU si disponible
            optimize_for_mobile: Optimiser pour appareils mobiles et systèmes embarqués
        """
        self.neurax_path = neurax_path
        self.training_path = training_path
        self.evaluation_path = evaluation_path
        self.test_path = test_path
        self.output_dir = output_dir
        self.use_gpu = use_gpu
        self.optimize_for_mobile = optimize_for_mobile
        
        # S'assurer que le répertoire de sortie existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Ajouter le chemin Neurax au path
        sys.path.append(neurax_path)
        
        # Importer les modules Neurax
        try:
            if optimize_for_mobile:
                from quantum_gravity_sim_mobile import QuantumGravitySimulatorMobile
                self.simulator_class = QuantumGravitySimulatorMobile
                logger.info("Simulateur mobile importé avec succès")
            else:
                try:
                    # Tenter d'importer la version GPU
                    from quantum_gravity_sim_gpu import QuantumGravitySimulatorGPU
                    self.simulator_class = QuantumGravitySimulatorGPU
                    logger.info("Simulateur GPU importé avec succès")
                except ImportError:
                    # Fallback sur la version CPU
                    from quantum_gravity_sim import QuantumGravitySimulator
                    self.simulator_class = QuantumGravitySimulator
                    logger.info("Simulateur CPU importé avec succès")
        except ImportError as e:
            logger.error(f"Erreur lors de l'importation des modules Neurax: {str(e)}")
            raise
        
        # Détection de l'accélération matérielle disponible
        self.detect_hardware_acceleration()
        
        # Création du simulateur
        self.create_simulator()
    
    def detect_hardware_acceleration(self):
        """
        Détecte les accélérateurs matériels disponibles dans l'environnement Kaggle
        """
        self.gpu_available = False
        self.tpu_available = False
        
        try:
            # Vérifier la disponibilité du GPU via PyTorch
            import torch
            self.gpu_available = torch.cuda.is_available()
            if self.gpu_available:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_count = torch.cuda.device_count()
                logger.info(f"GPU détecté via PyTorch: {gpu_name} (x{gpu_count})")
        except ImportError:
            pass
        
        try:
            # Vérifier la disponibilité du GPU via TensorFlow
            import tensorflow as tf
            if not self.gpu_available:
                self.gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
                if self.gpu_available:
                    gpus = tf.config.list_physical_devices('GPU')
                    logger.info(f"GPU détecté via TensorFlow: {len(gpus)} disponibles")
            
            # Vérifier la disponibilité du TPU
            tpus = tf.config.list_physical_devices('TPU')
            self.tpu_available = len(tpus) > 0
            if self.tpu_available:
                logger.info(f"TPU détecté via TensorFlow: {len(tpus)} disponibles")
        except ImportError:
            pass
        
        # Informations sur le CPU
        import multiprocessing
        self.cpu_count = multiprocessing.cpu_count()
        logger.info(f"Nombre de cœurs CPU: {self.cpu_count}")
        
        # Utiliser GPU uniquement si disponible et demandé
        self.use_gpu = self.use_gpu and self.gpu_available
        logger.info(f"Utilisation GPU: {'Oui' if self.use_gpu else 'Non'}")
    
    def create_simulator(self):
        """
        Crée une instance du simulateur Neurax optimisé
        """
        if self.optimize_for_mobile:
            # Paramètres optimisés pour mobile
            self.simulator = self.simulator_class(
                grid_size=16,
                time_steps=4,
                precision="float16",
                use_cache=True
            )
            logger.info("Simulateur mobile créé")
        else:
            # Paramètres standard
            if hasattr(self.simulator_class, 'use_gpu'):
                self.simulator = self.simulator_class(
                    grid_size=32,
                    time_steps=8,
                    use_gpu=self.use_gpu,
                    use_cache=True
                )
            else:
                self.simulator = self.simulator_class(
                    grid_size=32,
                    time_steps=8
                )
            logger.info("Simulateur standard créé")
    
    def load_arc_puzzle(self, puzzle_id: str, phase: str = "training") -> Dict[str, Any]:
        """
        Charge un puzzle ARC depuis les fichiers selon les standards de la compétition
        
        Args:
            puzzle_id: Identifiant du puzzle
            phase: Phase (training/evaluation/test)
            
        Returns:
            Données du puzzle
        """
        base_path = ""
        if phase == "training":
            base_path = self.training_path
        elif phase == "evaluation":
            base_path = self.evaluation_path
        elif phase == "test":
            base_path = self.test_path
        else:
            raise ValueError(f"Phase inconnue: {phase}")
        
        # Chemins selon spécifications de la compétition ARC-Prize-2025
        puzzle_path = os.path.join(base_path, f"{puzzle_id}.json")
        
        try:
            with open(puzzle_path, 'r') as f:
                puzzle_data = json.load(f)
            
            return {
                "id": puzzle_id,
                "train": puzzle_data["train"] if "train" in puzzle_data else [],
                "test": puzzle_data["test"] if "test" in puzzle_data else []
            }
        except FileNotFoundError:
            logger.warning(f"Puzzle {puzzle_id} non trouvé dans {phase}")
            return {"id": puzzle_id, "train": [], "test": []}
        except json.JSONDecodeError:
            logger.error(f"Erreur de décodage JSON pour {puzzle_id}")
            return {"id": puzzle_id, "train": [], "test": []}
    
    def get_puzzle_ids(self, phase: str = "training") -> List[str]:
        """
        Récupère la liste des identifiants des puzzles disponibles
        
        Args:
            phase: Phase (training/evaluation/test)
            
        Returns:
            Liste des identifiants de puzzles
        """
        base_path = ""
        if phase == "training":
            base_path = self.training_path
        elif phase == "evaluation":
            base_path = self.evaluation_path
        elif phase == "test":
            base_path = self.test_path
        else:
            raise ValueError(f"Phase inconnue: {phase}")
        
        try:
            # Lister tous les fichiers JSON
            puzzle_ids = []
            for filename in os.listdir(base_path):
                if filename.endswith('.json'):
                    puzzle_id = filename.split('.')[0]
                    puzzle_ids.append(puzzle_id)
            
            logger.info(f"Phase {phase}: {len(puzzle_ids)} puzzles trouvés")
            return puzzle_ids
        except FileNotFoundError:
            logger.error(f"Répertoire {base_path} non trouvé")
            return []
    
    def process_puzzle(self, puzzle_id: str, phase: str = "training") -> Dict[str, Any]:
        """
        Traite un puzzle ARC
        
        Args:
            puzzle_id: Identifiant du puzzle
            phase: Phase (training/evaluation/test)
            
        Returns:
            Résultats du traitement
        """
        start_time = time.time()
        
        # Charger le puzzle
        puzzle_data = self.load_arc_puzzle(puzzle_id, phase)
        
        # Vérifier que le puzzle contient des données
        if not puzzle_data.get("train") and not puzzle_data.get("test"):
            return {
                "puzzle_id": puzzle_id,
                "phase": phase,
                "status": "FAIL",
                "error": "Puzzle vide ou non trouvé",
                "duration": time.time() - start_time
            }
        
        # Traiter le puzzle avec le simulateur
        try:
            sim_results = self.simulator.process_puzzle(puzzle_data)
            
            # Extraire la prédiction
            prediction = sim_results.get("prediction", [])
            
            # Vérifier la solution si disponible (training/evaluation)
            status = "PASS"
            if phase != "test" and puzzle_data.get("test"):
                test_examples = puzzle_data["test"]
                if test_examples and "output" in test_examples[0]:
                    expected = test_examples[0]["output"]
                    # Comparer la prédiction avec la sortie attendue
                    if not np.array_equal(np.array(prediction), np.array(expected)):
                        status = "FAIL"
            
            return {
                "puzzle_id": puzzle_id,
                "phase": phase,
                "status": status,
                "prediction": prediction,
                "duration": time.time() - start_time
            }
        except Exception as e:
            logger.error(f"Erreur lors du traitement du puzzle {puzzle_id}: {str(e)}")
            return {
                "puzzle_id": puzzle_id,
                "phase": phase,
                "status": "FAIL",
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    def process_all_puzzles(self, phase: str = "training", max_puzzles: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Traite tous les puzzles d'une phase
        
        Args:
            phase: Phase à traiter
            max_puzzles: Nombre maximum de puzzles à traiter
            
        Returns:
            Liste des résultats
        """
        puzzle_ids = self.get_puzzle_ids(phase)
        
        if max_puzzles is not None and max_puzzles < len(puzzle_ids):
            puzzle_ids = puzzle_ids[:max_puzzles]
            logger.info(f"Limitation à {max_puzzles} puzzles pour la phase {phase}")
        
        results = []
        total = len(puzzle_ids)
        success_count = 0
        
        logger.info(f"Traitement de {total} puzzles de la phase {phase}")
        
        for i, puzzle_id in enumerate(puzzle_ids):
            # Traiter le puzzle
            result = self.process_puzzle(puzzle_id, phase)
            results.append(result)
            
            # Mise à jour des statistiques
            if result.get("status") == "PASS":
                success_count += 1
            
            # Afficher la progression
            if (i+1) % 10 == 0 or i == total-1:
                logger.info(f"Progression: {i+1}/{total} ({(i+1)/total*100:.1f}%) - "
                         f"Réussite: {success_count}/{i+1} ({success_count/(i+1)*100:.1f}%)")
        
        return results
    
    def prepare_kaggle_submission(self, results: Optional[List[Dict[str, Any]]] = None, 
                               phase: str = "test",
                               output_file: str = "submission.csv") -> None:
        """
        Prépare une soumission au format requis par la compétition Kaggle ARC-Prize-2025
        
        Args:
            results: Résultats pré-calculés (facultatif)
            phase: Phase pour laquelle préparer la soumission
            output_file: Fichier de sortie pour la soumission
        """
        logger.info(f"Préparation de la soumission Kaggle pour la phase {phase}")
        
        # Si les résultats ne sont pas fournis, les calculer
        if results is None:
            results = self.process_all_puzzles(phase)
        
        # Préparer le fichier de soumission selon le format de la compétition
        submission_path = os.path.join(self.output_dir, output_file)
        
        with open(submission_path, 'w') as f:
            # Écrire l'en-tête
            f.write("puzzle_id,output\n")
            
            for result in results:
                puzzle_id = result.get("puzzle_id", "")
                prediction = result.get("prediction", [])
                
                # Formater la prédiction en JSON
                prediction_json = json.dumps(prediction)
                
                # Écrire la ligne
                f.write(f"{puzzle_id},{prediction_json}\n")
        
        logger.info(f"Soumission enregistrée: {submission_path}")
    
    def export_results(self, results: List[Dict[str, Any]], phase: str = "training") -> None:
        """
        Exporte les résultats détaillés
        
        Args:
            results: Résultats à exporter
            phase: Phase concernée
        """
        # Fichier de résultats détaillés
        details_path = os.path.join(self.output_dir, f"arc_{phase}_results.json")
        with open(details_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Fichier de statistiques
        stats = self.calculate_statistics(results)
        stats_path = os.path.join(self.output_dir, f"arc_{phase}_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Résultats exportés: {details_path}, {stats_path}")
    
    def calculate_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calcule des statistiques sur les résultats
        
        Args:
            results: Liste des résultats
            
        Returns:
            Statistiques calculées
        """
        total = len(results)
        if total == 0:
            return {"total": 0, "success_rate": 0, "average_duration": 0}
        
        success_count = sum(1 for r in results if r.get("status") == "PASS")
        duration_sum = sum(r.get("duration", 0) for r in results)
        
        return {
            "total": total,
            "success_count": success_count,
            "success_rate": (success_count / total) * 100,
            "average_duration": duration_sum / total,
            "total_duration": duration_sum
        }
    
    def run_complete_workflow(self, training_max: Optional[int] = None,
                           evaluation_max: Optional[int] = None,
                           test_max: Optional[int] = None,
                           prepare_submission: bool = True) -> Dict[str, Any]:
        """
        Exécute le workflow complet: traitement des puzzles et préparation de la soumission
        
        Args:
            training_max: Nombre maximum de puzzles d'entraînement
            evaluation_max: Nombre maximum de puzzles d'évaluation
            test_max: Nombre maximum de puzzles de test
            prepare_submission: Préparer une soumission pour la phase de test
            
        Returns:
            Statistiques globales
        """
        all_stats = {}
        
        # Phase d'entraînement
        if training_max is not None:
            logger.info("=== PHASE D'ENTRAÎNEMENT ===")
            training_results = self.process_all_puzzles("training", training_max)
            self.export_results(training_results, "training")
            all_stats["training"] = self.calculate_statistics(training_results)
        
        # Phase d'évaluation
        if evaluation_max is not None:
            logger.info("=== PHASE D'ÉVALUATION ===")
            evaluation_results = self.process_all_puzzles("evaluation", evaluation_max)
            self.export_results(evaluation_results, "evaluation")
            all_stats["evaluation"] = self.calculate_statistics(evaluation_results)
        
        # Phase de test
        if test_max is not None:
            logger.info("=== PHASE DE TEST ===")
            test_results = self.process_all_puzzles("test", test_max)
            self.export_results(test_results, "test")
            all_stats["test"] = self.calculate_statistics(test_results)
            
            # Préparer la soumission pour Kaggle
            if prepare_submission:
                self.prepare_kaggle_submission(test_results)
        
        # Exporter les statistiques globales
        global_stats_path = os.path.join(self.output_dir, "arc_global_stats.json")
        with open(global_stats_path, 'w') as f:
            json.dump(all_stats, f, indent=2)
        logger.info(f"Statistiques globales exportées: {global_stats_path}")
        
        return all_stats


def test_mode_setup():
    """
    Configure l'environnement pour le mode test (hors Kaggle)
    """
    os.makedirs("./training", exist_ok=True)
    os.makedirs("./evaluation", exist_ok=True)
    os.makedirs("./test", exist_ok=True)
    
    # Créer un exemple de puzzle pour le test
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
    
    with open("./training/test_puzzle.json", 'w') as f:
        json.dump(test_puzzle, f)
    
    with open("./evaluation/test_puzzle.json", 'w') as f:
        json.dump(test_puzzle, f)
    
    with open("./test/test_puzzle.json", 'w') as f:
        json.dump(test_puzzle, f)
    
    logger.info("Mode test configuré avec puzzle exemple")


def main():
    """
    Point d'entrée principal
    """
    parser = argparse.ArgumentParser(description="Adaptateur Kaggle pour la compétition ARC-Prize-2025")
    parser.add_argument("--neurax-path", type=str, default=".", help="Chemin vers les modules Neurax")
    parser.add_argument("--training", type=str, default="./training", help="Chemin vers les données d'entraînement")
    parser.add_argument("--evaluation", type=str, default="./evaluation", help="Chemin vers les données d'évaluation")
    parser.add_argument("--test", type=str, default="./test", help="Chemin vers les données de test")
    parser.add_argument("--output", type=str, default="./output", help="Répertoire de sortie")
    parser.add_argument("--no-gpu", action="store_true", help="Désactiver l'utilisation du GPU")
    parser.add_argument("--mobile", action="store_true", help="Optimiser pour appareil mobile")
    parser.add_argument("--training-max", type=int, default=None, help="Nombre max de puzzles d'entraînement")
    parser.add_argument("--evaluation-max", type=int, default=None, help="Nombre max de puzzles d'évaluation")
    parser.add_argument("--test-max", type=int, default=None, help="Nombre max de puzzles de test")
    parser.add_argument("--test-mode", action="store_true", help="Exécuter en mode test")
    
    args = parser.parse_args()
    
    # Configuration du mode test si demandé ou si hors Kaggle
    if args.test_mode or TEST_MODE:
        test_mode_setup()
    
    # Créer l'adaptateur
    adapter = ArcPrizeAdapter(
        neurax_path=args.neurax_path,
        training_path=args.training,
        evaluation_path=args.evaluation,
        test_path=args.test,
        output_dir=args.output,
        use_gpu=not args.no_gpu,
        optimize_for_mobile=args.mobile
    )
    
    # Exécuter le workflow pour les phases spécifiées
    stats = {}
    
    if args.training_max is not None:
        logger.info("=== PHASE D'ENTRAÎNEMENT ===")
        training_results = adapter.process_all_puzzles("training", args.training_max)
        adapter.export_results(training_results, "training")
        stats["training"] = adapter.calculate_statistics(training_results)
    
    if args.evaluation_max is not None:
        logger.info("=== PHASE D'ÉVALUATION ===")
        evaluation_results = adapter.process_all_puzzles("evaluation", args.evaluation_max)
        adapter.export_results(evaluation_results, "evaluation")
        stats["evaluation"] = adapter.calculate_statistics(evaluation_results)
    
    if args.test_max is not None:
        logger.info("=== PHASE DE TEST ===")
        test_results = adapter.process_all_puzzles("test", args.test_max)
        adapter.export_results(test_results, "test")
        stats["test"] = adapter.calculate_statistics(test_results)
        
        # Préparer la soumission pour Kaggle
        adapter.prepare_kaggle_submission(test_results)
    
    # Exporter les statistiques globales
    global_stats_path = os.path.join(args.output, "arc_global_stats.json")
    with open(global_stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Statistiques globales exportées: {global_stats_path}")


if __name__ == "__main__":
    main()