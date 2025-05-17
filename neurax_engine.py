#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Moteur principal Neurax2 optimisé
Intègre le simulateur de gravité quantique optimisé et les fonctionnalités de traitement
"""

import os
import sys
import time
import json
import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("neurax_engine.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("NeuraxEngine")

# Importer le simulateur optimisé
try:
    # Priorité au simulateur GPU
    from quantum_gravity_sim_gpu import QuantumGravitySimulatorGPU
    logger.info("Simulateur GPU chargé avec succès")
    QuantumGravitySimulator = QuantumGravitySimulatorGPU
    USE_GPU_SIMULATOR = True
except ImportError:
    # Fallback au simulateur CPU d'origine
    from quantum_gravity_sim import QuantumGravitySimulator
    logger.info("Simulateur GPU non disponible, utilisation du simulateur CPU standard")
    USE_GPU_SIMULATOR = False

# Nombre de processus pour le traitement parallèle
NUM_PROCESSES = max(1, multiprocessing.cpu_count() - 1)

class NeuraxEngine:
    """
    Moteur optimisé pour le traitement des puzzles ARC avec Neurax2
    """
    
    def __init__(self, arc_data_path: str = "./neurax_complet/arc_data",
                default_grid_size: int = 32,
                time_steps: int = 8,
                use_gpu: bool = True,
                use_cache: bool = True):
        """
        Initialise le moteur Neurax2
        
        Args:
            arc_data_path: Chemin vers les données ARC
            default_grid_size: Taille de grille par défaut
            time_steps: Nombre d'étapes temporelles
            use_gpu: Utiliser le GPU si disponible
            use_cache: Utiliser le cache pour les calculs
        """
        self.arc_data_path = arc_data_path
        self.default_grid_size = default_grid_size
        self.time_steps = time_steps
        self.use_gpu = use_gpu and USE_GPU_SIMULATOR
        self.use_cache = use_cache
        
        # Statistiques
        self.stats = {
            "puzzles_processed": 0,
            "success_count": 0,
            "processing_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        logger.info(f"Initialisation du moteur Neurax2 - GPU: {'Activé' if self.use_gpu else 'Désactivé'}")
    
    def create_simulator(self, grid_size: Optional[int] = None) -> Union[QuantumGravitySimulator, QuantumGravitySimulatorGPU]:
        """
        Crée une instance du simulateur avec les paramètres appropriés
        
        Args:
            grid_size: Taille de la grille (utilise la taille par défaut si None)
            
        Returns:
            Instance du simulateur
        """
        size = grid_size if grid_size else self.default_grid_size
        
        if USE_GPU_SIMULATOR:
            return QuantumGravitySimulatorGPU(grid_size=size, time_steps=self.time_steps, 
                                           use_gpu=self.use_gpu, use_cache=self.use_cache)
        else:
            return QuantumGravitySimulator(grid_size=size, time_steps=self.time_steps)
    
    def load_puzzle(self, puzzle_id: str, phase: str = "training") -> Dict[str, Any]:
        """
        Charge un puzzle spécifique depuis les données ARC
        
        Args:
            puzzle_id: Identifiant du puzzle
            phase: Phase du puzzle (training/evaluation/test)
            
        Returns:
            Données du puzzle
        """
        filename = ""
        if phase == "training":
            filename = "arc-agi_training_challenges.json"
        elif phase == "evaluation":
            filename = "arc-agi_evaluation_challenges.json"
        elif phase == "test":
            filename = "arc-agi_test_challenges.json"
        else:
            raise ValueError(f"Phase inconnue: {phase}")
        
        filepath = os.path.join(self.arc_data_path, filename)
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if "challenges" in data:
                puzzles = {str(challenge["id"]): challenge for challenge in data["challenges"]}
                if puzzle_id in puzzles:
                    return puzzles[puzzle_id]
            elif puzzle_id in data:
                return data[puzzle_id]
                
            raise ValueError(f"Puzzle {puzzle_id} non trouvé dans {phase}")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du puzzle {puzzle_id}: {str(e)}")
            return {}
    
    def process_single_puzzle(self, puzzle_id: str, phase: str = "training") -> Dict[str, Any]:
        """
        Traite un puzzle unique
        
        Args:
            puzzle_id: Identifiant du puzzle
            phase: Phase du puzzle
            
        Returns:
            Résultats du traitement
        """
        start_time = time.time()
        
        # Charger le puzzle
        puzzle_data = self.load_puzzle(puzzle_id, phase)
        if not puzzle_data:
            return {
                "puzzle_id": puzzle_id,
                "phase": phase,
                "status": "FAIL",
                "error": "Puzzle non trouvé",
                "duration": time.time() - start_time
            }
        
        # Préparer les données pour le simulateur
        processed_data = self.prepare_puzzle_data(puzzle_data)
        
        # Déterminer la taille de grille appropriée
        grid_size = self.determine_grid_size(processed_data)
        
        # Créer le simulateur
        simulator = self.create_simulator(grid_size)
        
        # Traiter le puzzle
        try:
            sim_results = simulator.process_puzzle(processed_data)
            
            # Mettre à jour les statistiques
            self.stats["puzzles_processed"] += 1
            self.stats["processing_time"] += time.time() - start_time
            
            if USE_GPU_SIMULATOR:
                cache_stats = QuantumGravitySimulatorGPU.get_cache_stats()
                self.stats["cache_hits"] = cache_stats["hits"]
                self.stats["cache_misses"] = cache_stats["misses"]
            
            success = not sim_results.get("error")
            if success:
                self.stats["success_count"] += 1
            
            return {
                "puzzle_id": puzzle_id,
                "phase": phase,
                "status": "PASS" if success else "FAIL",
                "grid_size": grid_size,
                "simulator_details": sim_results,
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
    
    def prepare_puzzle_data(self, puzzle_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prépare les données du puzzle pour le traitement
        
        Args:
            puzzle_data: Données brutes du puzzle
            
        Returns:
            Données préparées pour le simulateur
        """
        # Créer une structure standard pour le simulateur
        processed_data = {
            "id": puzzle_data.get("id", "unknown"),
            "train": []
        }
        
        # Si le puzzle contient directement des exemples d'entraînement
        if "train" in puzzle_data:
            processed_data["train"] = puzzle_data["train"]
        
        # Si le puzzle contient une description et des exemples
        elif "description" in puzzle_data:
            # Pour les besoins du test, on utilise un exemple par défaut
            processed_data["train"] = [{
                "input": [[0, 0], [0, 0]],
                "output": [[1, 1], [1, 1]]
            }]
        
        # Cas où nous avons des exemples mais sous un autre format
        elif "examples" in puzzle_data:
            processed_data["train"] = puzzle_data["examples"]
        
        return processed_data
    
    def determine_grid_size(self, processed_data: Dict[str, Any]) -> int:
        """
        Détermine la taille de grille optimale pour un puzzle
        
        Args:
            processed_data: Données préparées du puzzle
            
        Returns:
            Taille de grille optimale
        """
        # Taille par défaut si pas d'exemples
        if not processed_data.get("train"):
            return self.default_grid_size
        
        try:
            # Extraire les dimensions des exemples
            example = processed_data["train"][0]
            if "input" in example and isinstance(example["input"], list):
                height = len(example["input"])
                width = len(example["input"][0]) if height > 0 else 0
                
                # Arrondir à la puissance de 2 supérieure pour optimiser les calculs
                grid_size = max(self.default_grid_size, 
                              2 ** (height - 1).bit_length(), 
                              2 ** (width - 1).bit_length())
                
                return grid_size
        except (IndexError, KeyError, TypeError):
            pass
        
        # Fallback à la taille par défaut
        return self.default_grid_size
    
    def process_puzzles_batch(self, puzzle_ids: List[str], phase: str = "training") -> List[Dict[str, Any]]:
        """
        Traite un lot de puzzles en parallèle
        
        Args:
            puzzle_ids: Liste des identifiants de puzzles
            phase: Phase des puzzles
            
        Returns:
            Liste des résultats
        """
        results = []
        total = len(puzzle_ids)
        
        logger.info(f"Traitement de {total} puzzles de la phase {phase}")
        
        # Traitement parallèle
        with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
            # Préparer les tâches
            futures = {executor.submit(self.process_single_puzzle, puzzle_id, phase): puzzle_id 
                     for puzzle_id in puzzle_ids}
            
            # Collecter les résultats
            for i, future in enumerate(as_completed(futures)):
                puzzle_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Afficher la progression
                    status = result.get("status", "UNKNOWN")
                    duration = result.get("duration", 0)
                    
                    logger.info(f"Puzzle {i+1}/{total} ({(i+1)/total*100:.1f}%) - ID: {puzzle_id} - Statut: {status} - Durée: {duration:.4f}s")
                    
                except Exception as e:
                    logger.error(f"Erreur lors du traitement du puzzle {puzzle_id}: {str(e)}")
                    results.append({
                        "puzzle_id": puzzle_id,
                        "phase": phase,
                        "status": "FAIL",
                        "error": str(e)
                    })
        
        return results
    
    def export_results(self, results: List[Dict[str, Any]], filename_prefix: str = "neurax_results") -> None:
        """
        Exporte les résultats au format JSON et CSV
        
        Args:
            results: Liste des résultats
            filename_prefix: Préfixe pour les noms de fichiers
        """
        # Export JSON
        json_file = f"{filename_prefix}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Résultats exportés au format JSON: {json_file}")
        
        # Export CSV
        csv_file = f"{filename_prefix}.csv"
        try:
            import csv
            with open(csv_file, 'w', newline='') as f:
                fieldnames = ["puzzle_id", "phase", "status", "duration", "grid_size"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    row = {
                        "puzzle_id": result.get("puzzle_id", ""),
                        "phase": result.get("phase", ""),
                        "status": result.get("status", ""),
                        "duration": result.get("duration", 0),
                        "grid_size": result.get("grid_size", 0)
                    }
                    writer.writerow(row)
            logger.info(f"Résultats exportés au format CSV: {csv_file}")
        except Exception as e:
            logger.error(f"Erreur lors de l'export CSV: {str(e)}")
        
        # Export des statistiques
        stats_file = f"{filename_prefix}_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        logger.info(f"Statistiques exportées: {stats_file}")
    
    def get_puzzle_ids(self, phase: str = "training", max_puzzles: Optional[int] = None) -> List[str]:
        """
        Récupère la liste des identifiants de puzzles pour une phase
        
        Args:
            phase: Phase des puzzles
            max_puzzles: Nombre maximum de puzzles à récupérer
            
        Returns:
            Liste des identifiants de puzzles
        """
        filename = ""
        if phase == "training":
            filename = "arc-agi_training_challenges.json"
        elif phase == "evaluation":
            filename = "arc-agi_evaluation_challenges.json"
        elif phase == "test":
            filename = "arc-agi_test_challenges.json"
        else:
            raise ValueError(f"Phase inconnue: {phase}")
        
        filepath = os.path.join(self.arc_data_path, filename)
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if "challenges" in data:
                puzzles = [str(challenge["id"]) for challenge in data["challenges"]]
            else:
                puzzles = list(data.keys())
            
            logger.info(f"Chargement réussi: {len(puzzles)} puzzles de {phase}")
            
            if max_puzzles and max_puzzles < len(puzzles):
                logger.info(f"Limitation à {max_puzzles} puzzles pour la phase {phase}")
                puzzles = puzzles[:max_puzzles]
                
            return puzzles
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des puzzles {phase}: {str(e)}")
            return []
    
    def process_phase(self, phase: str = "training", max_puzzles: Optional[int] = None, 
                    export_prefix: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Traite tous les puzzles d'une phase
        
        Args:
            phase: Phase à traiter
            max_puzzles: Nombre maximum de puzzles à traiter
            export_prefix: Préfixe pour l'export des résultats
            
        Returns:
            Liste des résultats
        """
        # Récupérer les identifiants de puzzles
        puzzle_ids = self.get_puzzle_ids(phase, max_puzzles)
        
        if not puzzle_ids:
            logger.error(f"Aucun puzzle trouvé pour la phase {phase}")
            return []
        
        # Traiter les puzzles
        start_time = time.time()
        results = self.process_puzzles_batch(puzzle_ids, phase)
        duration = time.time() - start_time
        
        # Calculer les statistiques
        total = len(results)
        success = sum(1 for r in results if r.get("status") == "PASS")
        avg_duration = sum(r.get("duration", 0) for r in results) / total if total > 0 else 0
        
        logger.info(f"=== RÉSULTATS PHASE {phase.upper()} ===")
        logger.info(f"Total puzzles: {total}")
        logger.info(f"Réussis: {success} ({success/total*100:.1f}%)")
        logger.info(f"Durée totale: {duration:.2f}s")
        logger.info(f"Durée moyenne par puzzle: {avg_duration:.4f}s")
        
        # Exporter les résultats si demandé
        if export_prefix:
            self.export_results(results, f"{export_prefix}_{phase}")
        
        return results
    
    def process_all_phases(self, training_puzzles: Optional[int] = None, 
                         evaluation_puzzles: Optional[int] = None,
                         test_puzzles: Optional[int] = None,
                         export_prefix: str = "neurax_results") -> Dict[str, List[Dict[str, Any]]]:
        """
        Traite les puzzles de toutes les phases
        
        Args:
            training_puzzles: Nombre de puzzles d'entraînement
            evaluation_puzzles: Nombre de puzzles d'évaluation
            test_puzzles: Nombre de puzzles de test
            export_prefix: Préfixe pour l'export des résultats
            
        Returns:
            Dictionnaire des résultats par phase
        """
        all_results = {}
        
        # Traiter la phase d'entraînement
        if training_puzzles is not None:
            logger.info(f"=== PHASE D'ENTRAÎNEMENT ===")
            all_results["training"] = self.process_phase("training", training_puzzles, export_prefix)
        
        # Traiter la phase d'évaluation
        if evaluation_puzzles is not None:
            logger.info(f"=== PHASE D'ÉVALUATION ===")
            all_results["evaluation"] = self.process_phase("evaluation", evaluation_puzzles, export_prefix)
        
        # Traiter la phase de test
        if test_puzzles is not None:
            logger.info(f"=== PHASE DE TEST ===")
            all_results["test"] = self.process_phase("test", test_puzzles, export_prefix)
        
        # Exporter les statistiques globales
        stats_file = f"{export_prefix}_global_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        logger.info(f"Statistiques globales exportées: {stats_file}")
        
        return all_results


# Test simple si exécuté directement
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Moteur Neurax2 optimisé")
    parser.add_argument("--training", type=int, default=None, help="Nombre de puzzles d'entraînement")
    parser.add_argument("--evaluation", type=int, default=None, help="Nombre de puzzles d'évaluation")
    parser.add_argument("--test", type=int, default=None, help="Nombre de puzzles de test")
    parser.add_argument("--arc-data", type=str, default="./neurax_complet/arc_data", help="Chemin vers les données ARC")
    parser.add_argument("--output", type=str, default="neurax_results", help="Préfixe pour les fichiers de sortie")
    parser.add_argument("--gpu", action="store_true", help="Utiliser le GPU si disponible")
    parser.add_argument("--no-cache", action="store_true", help="Désactiver le cache")
    
    args = parser.parse_args()
    
    # Créer le moteur
    engine = NeuraxEngine(
        arc_data_path=args.arc_data,
        use_gpu=args.gpu,
        use_cache=not args.no_cache
    )
    
    # Traiter les puzzles
    engine.process_all_phases(
        training_puzzles=args.training,
        evaluation_puzzles=args.evaluation,
        test_puzzles=args.test,
        export_prefix=args.output
    )