#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script d'adaptation de Neurax2 pour l'environnement Kaggle et l'API ARC-Prize-2025
Ce script sert d'interface entre le système Neurax2 et les notebooks Kaggle
"""

import os
import sys
import json
import logging
import time
import argparse
from typing import Dict, List, Any, Optional, Union
import numpy as np

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("neurax_kaggle.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("KaggleArcAdapter")

# Détection automatique de l'environnement Kaggle
IN_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

class KaggleArcAdapter:
    """
    Adaptateur pour l'exécution de Neurax2 sur Kaggle avec les API ARC-Prize-2025
    """
    
    def __init__(self, 
                neurax_engine_path: str = ".",
                arc_data_path: str = "./arc_data",
                use_gpu: bool = True,
                use_cache: bool = True,
                output_format: str = "arc_prize_2025"):
        """
        Initialise l'adaptateur Kaggle
        
        Args:
            neurax_engine_path: Chemin vers le module du moteur Neurax2
            arc_data_path: Chemin vers les données ARC
            use_gpu: Utiliser le GPU si disponible
            use_cache: Utiliser le cache pour les calculs
            output_format: Format de sortie pour la soumission
        """
        self.neurax_engine_path = neurax_engine_path
        self.arc_data_path = arc_data_path
        self.use_gpu = use_gpu
        self.use_cache = use_cache
        self.output_format = output_format
        
        # Importation dynamique du moteur Neurax
        sys.path.append(neurax_engine_path)
        try:
            from neurax_engine import NeuraxEngine
            self.engine_class = NeuraxEngine
            logger.info("NeuraxEngine importé avec succès")
        except ImportError as e:
            logger.error(f"Erreur lors de l'importation du moteur Neurax: {str(e)}")
            raise
        
        # Détection du matériel disponible
        self.detect_hardware()
        
        # Instance du moteur (créé lors du premier appel)
        self._engine = None
    
    def detect_hardware(self):
        """
        Détecte le matériel disponible sur Kaggle
        """
        try:
            import torch
            self.gpu_available = torch.cuda.is_available()
            if self.gpu_available:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_count = torch.cuda.device_count()
                logger.info(f"GPU détecté: {gpu_name} (x{gpu_count})")
            else:
                logger.info("Aucun GPU détecté via PyTorch")
        except ImportError:
            try:
                import tensorflow as tf
                self.gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
                if self.gpu_available:
                    gpus = tf.config.list_physical_devices('GPU')
                    logger.info(f"GPU détecté via TensorFlow: {len(gpus)} disponibles")
                else:
                    logger.info("Aucun GPU détecté via TensorFlow")
            except ImportError:
                self.gpu_available = False
                logger.info("Ni PyTorch ni TensorFlow ne sont disponibles pour la détection GPU")
        
        # Informations sur le CPU
        import multiprocessing
        self.cpu_count = multiprocessing.cpu_count()
        logger.info(f"Nombre de cœurs CPU: {self.cpu_count}")
        
        # Informations sur la mémoire
        import psutil
        self.total_memory = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # GB
        logger.info(f"Mémoire totale: {self.total_memory:.2f} GB")
    
    @property
    def engine(self):
        """
        Accès lazy-loading au moteur Neurax
        """
        if self._engine is None:
            self._engine = self.engine_class(
                arc_data_path=self.arc_data_path,
                use_gpu=self.use_gpu and self.gpu_available,
                use_cache=self.use_cache
            )
            logger.info("Moteur Neurax initialisé")
        return self._engine
    
    def optimize_for_mobile(self, 
                          min_grid_size: int = 8, 
                          max_grid_size: int = 32,
                          use_quantization: bool = True):
        """
        Optimise les paramètres du moteur pour les appareils mobiles et systèmes embarqués
        
        Args:
            min_grid_size: Taille minimale de grille à utiliser
            max_grid_size: Taille maximale de grille à utiliser
            use_quantization: Utiliser la quantification pour réduire l'empreinte mémoire
        """
        if hasattr(self.engine, 'default_grid_size'):
            original_size = self.engine.default_grid_size
            self.engine.default_grid_size = min(max(min_grid_size, 8), max_grid_size)
            logger.info(f"Taille de grille optimisée pour appareils mobiles: {original_size} -> {self.engine.default_grid_size}")
        
        # TODO: Implémenter la quantification complète quand le simulateur supportera cette fonctionnalité
        if use_quantization:
            logger.info("Optimisation par quantification activée pour les appareils mobiles")
            # Code de quantification à ajouter ici
    
    def prepare_kaggle_submission(self, output_file: str = "submission.json") -> None:
        """
        Prépare une soumission au format requis par la compétition ARC-Prize-2025
        
        Args:
            output_file: Fichier de sortie pour la soumission
        """
        logger.info("Préparation de la soumission pour la compétition ARC-Prize-2025")
        
        # Traitement des puzzles de test
        test_results = self.engine.process_phase("test", export_prefix=None)
        
        # Formater les résultats selon le format requis
        submission = {}
        for result in test_results:
            puzzle_id = result.get("puzzle_id")
            if puzzle_id and result.get("status") == "PASS":
                # Extraire la prédiction du résultat
                prediction = result.get("simulator_details", {}).get("prediction", [])
                if prediction:
                    submission[puzzle_id] = prediction
        
        # Sauvegarder la soumission
        with open(output_file, 'w') as f:
            json.dump(submission, f, indent=2)
        
        logger.info(f"Soumission générée avec {len(submission)} puzzles: {output_file}")
    
    def run_kaggle_benchmark(self, num_puzzles: int = 10) -> Dict[str, Any]:
        """
        Exécute un benchmark sur Kaggle pour mesurer les performances
        
        Args:
            num_puzzles: Nombre de puzzles à traiter pour le benchmark
            
        Returns:
            Résultats du benchmark
        """
        logger.info(f"Exécution d'un benchmark Kaggle sur {num_puzzles} puzzles")
        
        start_time = time.time()
        
        # Exécution sur chaque phase
        phases = ["training", "evaluation", "test"]
        results = {}
        
        for phase in phases:
            phase_start = time.time()
            phase_results = self.engine.process_phase(phase, max_puzzles=num_puzzles)
            phase_duration = time.time() - phase_start
            
            # Calculer les métriques
            successful = sum(1 for r in phase_results if r.get("status") == "PASS")
            success_rate = (successful / len(phase_results)) * 100 if phase_results else 0
            avg_duration = sum(r.get("duration", 0) for r in phase_results) / len(phase_results) if phase_results else 0
            
            results[phase] = {
                "total_puzzles": len(phase_results),
                "successful": successful,
                "success_rate": success_rate,
                "total_duration": phase_duration,
                "avg_duration": avg_duration
            }
            
            logger.info(f"Phase {phase}: {success_rate:.1f}% de réussite, {avg_duration:.4f}s par puzzle")
        
        # Résultats globaux
        total_duration = time.time() - start_time
        results["global"] = {
            "total_duration": total_duration,
            "hardware": {
                "gpu_available": self.gpu_available,
                "cpu_count": self.cpu_count,
                "total_memory_gb": self.total_memory
            }
        }
        
        logger.info(f"Benchmark terminé en {total_duration:.2f}s")
        return results
    
    def process_from_kaggle_api(self, kaggle_api_key: Optional[str] = None, 
                             dataset_slug: str = "arc-prize-2025") -> None:
        """
        Traite les puzzles en utilisant l'API Kaggle pour accéder aux données
        
        Args:
            kaggle_api_key: Clé API Kaggle (si None, utilise les variables d'environnement)
            dataset_slug: Slug du dataset Kaggle pour les puzzles ARC
        """
        if not IN_KAGGLE:
            if kaggle_api_key is None:
                kaggle_api_key = os.environ.get("KAGGLE_API_KEY")
            
            if kaggle_api_key is None:
                raise ValueError("Clé API Kaggle non fournie et non disponible dans les variables d'environnement")
            
            # Configurer l'API Kaggle
            os.environ["KAGGLE_API_KEY"] = kaggle_api_key
        
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            logger.info(f"Téléchargement des données depuis Kaggle: {dataset_slug}")
            
            # Télécharger les données
            api.dataset_download_files(dataset_slug, path=self.arc_data_path, unzip=True)
            logger.info("Données téléchargées et extraites avec succès")
            
            # Traiter les puzzles
            self.prepare_kaggle_submission()
            
        except ImportError:
            logger.error("Module Kaggle API non disponible. Installez-le avec: pip install kaggle")
            raise
        except Exception as e:
            logger.error(f"Erreur lors de l'utilisation de l'API Kaggle: {str(e)}")
            raise


def main():
    """
    Point d'entrée principal
    """
    parser = argparse.ArgumentParser(description="Adaptateur Neurax2 pour Kaggle et ARC-Prize-2025")
    parser.add_argument("--arc-data", type=str, default="./arc_data", help="Chemin vers les données ARC")
    parser.add_argument("--engine-path", type=str, default=".", help="Chemin vers le module du moteur Neurax2")
    parser.add_argument("--output", type=str, default="submission.json", help="Fichier de sortie pour la soumission")
    parser.add_argument("--benchmark", action="store_true", help="Exécuter un benchmark")
    parser.add_argument("--num-puzzles", type=int, default=10, help="Nombre de puzzles pour le benchmark")
    parser.add_argument("--mobile-optimize", action="store_true", help="Optimiser pour appareils mobiles")
    parser.add_argument("--min-grid", type=int, default=8, help="Taille minimale de grille pour les appareils mobiles")
    parser.add_argument("--max-grid", type=int, default=32, help="Taille maximale de grille pour les appareils mobiles")
    parser.add_argument("--no-gpu", action="store_true", help="Désactiver l'utilisation du GPU")
    parser.add_argument("--no-cache", action="store_true", help="Désactiver l'utilisation du cache")
    parser.add_argument("--kaggle-api", action="store_true", help="Utiliser l'API Kaggle pour les données")
    
    args = parser.parse_args()
    
    # Créer l'adaptateur
    adapter = KaggleArcAdapter(
        neurax_engine_path=args.engine_path,
        arc_data_path=args.arc_data,
        use_gpu=not args.no_gpu,
        use_cache=not args.no_cache
    )
    
    # Optimisation pour appareils mobiles si demandée
    if args.mobile_optimize:
        adapter.optimize_for_mobile(
            min_grid_size=args.min_grid,
            max_grid_size=args.max_grid
        )
    
    # Exécuter un benchmark si demandé
    if args.benchmark:
        results = adapter.run_kaggle_benchmark(num_puzzles=args.num_puzzles)
        benchmark_file = "kaggle_benchmark_results.json"
        with open(benchmark_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Résultats du benchmark sauvegardés: {benchmark_file}")
    
    # Utiliser l'API Kaggle si demandé
    if args.kaggle_api:
        adapter.process_from_kaggle_api()
    else:
        # Préparer une soumission standard
        adapter.prepare_kaggle_submission(output_file=args.output)


if __name__ == "__main__":
    main()