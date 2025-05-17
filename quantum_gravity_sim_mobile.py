#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simulateur de gravité quantique optimisé pour appareils mobiles et systèmes embarqués
Version optimisée en empreinte mémoire et performances pour fonctionner sur des appareils à ressources limitées
"""

import os
import sys
import time
import json
import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("quantum_sim_mobile.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("QuantumSimMobile")

class QuantumGravitySimulatorMobile:
    """
    Version optimisée du simulateur de gravité quantique pour appareils mobiles
    Caractéristiques:
    - Empreinte mémoire réduite
    - Calculs optimisés pour processeurs mobiles
    - Support de la quantification (float16/int8)
    - Adaptabilité dynamique aux ressources disponibles
    """
    
    # Cache statique partagé entre toutes les instances
    _result_cache = {}
    _cache_hits = 0
    _cache_misses = 0
    
    # Types de données supportés avec leur empreinte mémoire relative
    DTYPES = {
        "float32": {"numpy": np.float32, "relative_size": 1.0},
        "float16": {"numpy": np.float16, "relative_size": 0.5},
        "int8": {"numpy": np.int8, "relative_size": 0.25}
    }
    
    def __init__(self, grid_size: int = 16, 
                time_steps: int = 4,
                use_cache: bool = True,
                precision: str = "float32",
                memory_limit_mb: Optional[int] = None):
        """
        Initialise le simulateur de gravité quantique optimisé pour mobile
        
        Args:
            grid_size: Taille de la grille de simulation (NxN)
            time_steps: Nombre d'étapes temporelles de la simulation
            use_cache: Utiliser le cache de résultats pour éviter les recalculs
            precision: Précision des calculs ('float32', 'float16', 'int8')
            memory_limit_mb: Limite mémoire en MB (détection auto si None)
        """
        # Validation de la précision
        if precision not in self.DTYPES:
            logger.warning(f"Précision {precision} non supportée, utilisation de float32")
            precision = "float32"
        
        self.precision = precision
        self.dtype = self.DTYPES[precision]["numpy"]
        
        # Détection de la mémoire disponible
        if memory_limit_mb is None:
            try:
                import psutil
                available_memory = psutil.virtual_memory().available / (1024 * 1024)
                memory_limit_mb = int(available_memory * 0.5)  # Utiliser 50% max
                logger.info(f"Mémoire disponible détectée: {available_memory:.1f} MB, limite: {memory_limit_mb} MB")
            except ImportError:
                memory_limit_mb = 200  # Valeur par défaut conservative
                logger.info(f"Module psutil non disponible, limite mémoire par défaut: {memory_limit_mb} MB")
        
        self.memory_limit_mb = memory_limit_mb
        
        # Ajuster la taille de grille en fonction de la mémoire disponible
        adjusted_grid_size = self.adjust_grid_size(grid_size, time_steps, memory_limit_mb)
        if adjusted_grid_size != grid_size:
            logger.warning(f"Taille de grille ajustée pour respecter la limite mémoire: {grid_size} -> {adjusted_grid_size}")
            grid_size = adjusted_grid_size
        
        self.grid_size = grid_size
        self.time_steps = time_steps
        self.use_cache = use_cache
        
        # Paramètres optimisés pour appareils mobiles
        self.optimize_parameters()
        
        logger.info(f"Initialisation du simulateur mobile {grid_size}x{grid_size}x{time_steps} - Précision: {precision}")
    
    def adjust_grid_size(self, requested_size: int, time_steps: int, memory_limit_mb: int) -> int:
        """
        Ajuste la taille de grille pour respecter la limite mémoire
        
        Args:
            requested_size: Taille de grille demandée
            time_steps: Nombre d'étapes temporelles
            memory_limit_mb: Limite mémoire en MB
            
        Returns:
            Taille de grille ajustée
        """
        # Calculer la mémoire requise pour la taille demandée
        precision_factor = self.DTYPES[self.precision]["relative_size"]
        memory_needed_mb = (requested_size * requested_size * time_steps * 4 * precision_factor) / (1024 * 1024)
        
        # Si dans les limites, retourner la taille demandée
        if memory_needed_mb <= memory_limit_mb:
            return requested_size
        
        # Sinon, calculer la taille maximale possible
        max_size = int(np.sqrt((memory_limit_mb * 1024 * 1024) / (time_steps * 4 * precision_factor)))
        
        # Arrondir à la puissance de 2 inférieure pour optimiser les calculs
        return 2 ** (max_size - 1).bit_length() if max_size > 1 else 2
    
    def optimize_parameters(self):
        """
        Optimise les paramètres internes du simulateur pour les appareils mobiles
        """
        # Paramètres adaptés aux petites grilles et faible précision
        self.quantum_coupling = 0.8
        self.relativity_factor = 0.2
        self.fluctuation_scale = 1.0
        
        # Réduire le nombre d'itérations pour les appareils mobiles
        self.max_iterations = 3
        
        logger.info("Paramètres optimisés pour appareil mobile")
    
    @staticmethod
    def get_cache_stats() -> Dict[str, int]:
        """
        Retourne les statistiques du cache
        
        Returns:
            Statistiques du cache
        """
        return {
            "hits": QuantumGravitySimulatorMobile._cache_hits,
            "misses": QuantumGravitySimulatorMobile._cache_misses,
            "size": len(QuantumGravitySimulatorMobile._result_cache),
            "memory_usage": sum(sys.getsizeof(v) for v in QuantumGravitySimulatorMobile._result_cache.values())
        }
    
    @staticmethod
    def clear_cache():
        """
        Efface le cache partagé
        """
        QuantumGravitySimulatorMobile._result_cache.clear()
        logger.info("Cache effacé")
    
    def get_cache_key(self, intensity: float) -> str:
        """
        Génère une clé de cache unique pour les paramètres donnés
        
        Args:
            intensity: Intensité de la fluctuation quantique
            
        Returns:
            Clé de cache
        """
        return f"{self.grid_size}_{self.time_steps}_{intensity:.6f}_{self.precision}"
    
    def apply_quantum_fluctuations(self, intensity: float = 1.0) -> np.ndarray:
        """
        Applique des fluctuations quantiques sur l'espace-temps
        Version optimisée pour appareils mobiles
        
        Args:
            intensity: Intensité des fluctuations (1.0 = standard)
            
        Returns:
            Matrice d'espace-temps après fluctuations
        """
        logger.info(f"Application de fluctuations quantiques (intensité: {intensity:.2f})")
        
        # Vérification du cache
        cache_key = self.get_cache_key(intensity)
        if self.use_cache and cache_key in self._result_cache:
            QuantumGravitySimulatorMobile._cache_hits += 1
            return self._result_cache[cache_key].copy()
        
        QuantumGravitySimulatorMobile._cache_misses += 1
        
        # Initialisation de l'espace-temps avec précision réduite
        space_time = np.zeros((self.time_steps, self.grid_size, self.grid_size), dtype=self.dtype)
        
        # Générer une perturbation initiale simplifiée (optimisée pour mobile)
        perturbation = np.random.random((self.grid_size, self.grid_size)).astype(self.dtype) * intensity
        
        # Initialiser la première étape temporelle
        space_time[0] = perturbation
        
        # Propager les effets dans le temps de manière simplifiée
        for t in range(1, self.time_steps):
            # Copie de la couche précédente
            space_time_copy = space_time.copy()
            
            # Propagation simplifiée pour appareils mobiles
            # Pas de FFT, utilisation de convolutions simplifiées
            space_time[t] = (
                self.quantum_coupling * space_time_copy[t-1] + 
                self.relativity_factor * np.roll(space_time_copy[t-1], 1, axis=0)
            )
        
        # Normaliser pour avoir des valeurs entre 0 et 1
        space_time = (space_time - space_time.min()) / (space_time.max() - space_time.min() + 1e-8)
        
        # Stocker dans le cache si activé
        if self.use_cache:
            self._result_cache[cache_key] = space_time.copy()
        
        return space_time
    
    def run_simulation_step(self) -> Dict[str, Any]:
        """
        Exécute une étape de simulation quantum-gravitationnelle
        Version optimisée pour appareils mobiles
        
        Returns:
            Résultats de la simulation
        """
        logger.info("Exécution d'une étape de simulation")
        
        # Paramètres calibrés pour système mobile
        intensity = 1.5
        
        # Appliquer les fluctuations quantiques
        space_time = self.apply_quantum_fluctuations(intensity=intensity)
        
        # Extraire les résultats (version simplifiée pour mobile)
        result_layer = space_time[-1]  # Dernière couche temporelle
        
        # Quantifier les résultats pour réduire l'empreinte mémoire
        if self.precision == "int8":
            # Convertir directement en catégories (0-3)
            categories = 4  # Nombre de catégories simplifié
            result_matrix = (result_layer * categories).astype(np.int8)
        elif self.precision == "float16":
            # Arrondir à la précision float16
            result_matrix = result_layer.astype(np.float16)
        else:
            result_matrix = result_layer
        
        return {
            "space_time": space_time.astype(self.dtype),
            "result_matrix": result_matrix,
            "intensity": intensity,
            "grid_size": self.grid_size,
            "time_steps": self.time_steps,
            "precision": self.precision
        }
    
    def process_puzzle(self, puzzle_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Traite un puzzle ARC avec le simulateur optimisé pour mobile
        
        Args:
            puzzle_data: Données du puzzle au format ARC
            
        Returns:
            Résultats du traitement
        """
        try:
            # Exécuter la simulation
            sim_results = self.run_simulation_step()
            
            # Traiter les entrées/sorties du puzzle (version simplifiée)
            if "train" in puzzle_data and puzzle_data["train"]:
                example = puzzle_data["train"][0]
                if "input" in example and "output" in example:
                    # Transformation simple pour démonstration
                    result_matrix = sim_results["result_matrix"]
                    
                    # Créer une prédiction en adaptant les dimensions
                    input_grid = np.array(example["input"])
                    output_grid = np.array(example["output"])
                    
                    input_height, input_width = input_grid.shape
                    output_height, output_width = output_grid.shape
                    
                    # Redimensionner la sortie
                    prediction = result_matrix[:output_height, :output_width]
                    
                    # Convertir en liste Python pour compatibilité JSON
                    prediction_list = prediction.tolist()
                    
                    return {
                        "prediction": prediction_list,
                        "simulation_params": {
                            "grid_size": self.grid_size,
                            "time_steps": self.time_steps,
                            "precision": self.precision,
                            "memory_usage_mb": self.get_memory_usage_mb()
                        }
                    }
            
            # Cas par défaut
            return {
                "error": "Format de puzzle non supporté",
                "simulation_params": {
                    "grid_size": self.grid_size,
                    "time_steps": self.time_steps,
                    "precision": self.precision
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement du puzzle: {str(e)}")
            return {"error": str(e)}
    
    def get_memory_usage_mb(self) -> float:
        """
        Estime l'utilisation mémoire actuelle du simulateur
        
        Returns:
            Utilisation mémoire en MB
        """
        # Calculer l'empreinte mémoire approximative
        precision_factor = self.DTYPES[self.precision]["relative_size"]
        space_time_size = (self.grid_size * self.grid_size * self.time_steps * 4 * precision_factor) / (1024 * 1024)
        
        # Ajouter l'empreinte du cache
        cache_size_mb = sum(sys.getsizeof(v) for v in self._result_cache.values()) / (1024 * 1024)
        
        return space_time_size + cache_size_mb


# Test simple si exécuté directement
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simulateur de gravité quantique optimisé pour mobile")
    parser.add_argument("--grid-size", type=int, default=16, help="Taille de la grille")
    parser.add_argument("--time-steps", type=int, default=4, help="Nombre d'étapes temporelles")
    parser.add_argument("--memory-limit", type=int, default=None, help="Limite mémoire en MB")
    parser.add_argument("--precision", type=str, choices=["float32", "float16", "int8"], default="float16", 
                      help="Précision des calculs")
    parser.add_argument("--no-cache", action="store_true", help="Désactiver le cache")
    
    args = parser.parse_args()
    
    # Créer le simulateur
    simulator = QuantumGravitySimulatorMobile(
        grid_size=args.grid_size,
        time_steps=args.time_steps,
        use_cache=not args.no_cache,
        precision=args.precision,
        memory_limit_mb=args.memory_limit
    )
    
    # Exécuter la simulation
    start_time = time.time()
    results = simulator.run_simulation_step()
    duration = time.time() - start_time
    
    logger.info(f"Simulation exécutée en {duration:.4f}s")
    
    # Afficher les statistiques de mémoire
    memory_usage = simulator.get_memory_usage_mb()
    logger.info(f"Utilisation mémoire: {memory_usage:.2f} MB")
    
    # Afficher les statistiques du cache
    cache_stats = QuantumGravitySimulatorMobile.get_cache_stats()
    logger.info(f"Statistiques du cache: {cache_stats}")
    
    # Exporter les résultats
    result_matrix = results["result_matrix"]
    logger.info(f"Matrice de résultats: {result_matrix.shape}, type: {result_matrix.dtype}")
    
    # Tester avec un puzzle simple
    test_puzzle = {
        "train": [
            {
                "input": [[0, 0], [0, 0]],
                "output": [[1, 1], [1, 1]]
            }
        ]
    }
    
    puzzle_results = simulator.process_puzzle(test_puzzle)
    logger.info(f"Résultats du puzzle: {json.dumps(puzzle_results, indent=2)}")