#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Système d'Apprentissage et de Résolution pour les Puzzles ARC
Intégré avec le Réseau Neuronal Gravitationnel Quantique (Neurax)

Ce module implémente un système complet pour l'apprentissage et la résolution
des puzzles ARC en utilisant le simulateur Neurax et des techniques d'apprentissage
adaptées au raisonnement abstrait.
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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import traceback
import h5py
import pandas as pd
from tqdm import tqdm
import random
import copy
import warnings

# Import du simulateur de gravité quantique
from quantum_gravity_sim import QuantumGravitySimulator

# Constants
QUANTUM_GRID_SIZE = 50  # Taille de la grille pour le simulateur
NUM_TIME_STEPS = 10     # Nombre de pas temporels pour la simulation
MAX_ITERATIONS = 100    # Nombre maximum d'itérations d'apprentissage

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("arc_learning.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("ARCLearning")

# Assurez-vous que les chemins d'import sont corrects
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import des modules Neurax
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
        
    MODULE_IMPORT_SUCCESS = True
except Exception as e:
    logger.error(f"Erreur d'importation des modules Neurax: {str(e)}")
    traceback.print_exc()
    MODULE_IMPORT_SUCCESS = False


# Constantes de configuration
MAX_GRID_SIZE = 30  # Taille maximale des grilles ARC
QUANTUM_GRID_SIZE = 64  # Taille de la grille quantique
MAX_ITERATIONS = 1000  # Nombre maximal d'itérations d'apprentissage par puzzle
LEARNING_CONVERGENCE_THRESHOLD = 0.95  # Seuil de convergence pour l'apprentissage (précision)
NUM_TIME_STEPS = 16  # Nombre de pas temporels dans la simulation
MAX_PATTERNS = 200  # Nombre maximal de modèles de transformation à tester


class TransformationPatternBase:
    """Classe de base pour les patterns de transformation abstraits"""
    
    def __init__(self, name):
        self.name = name
        self.complexity = 1.0  # Complexité du pattern (1.0 = base)
        self.applicability_score = 0.0  # Score d'applicabilité (calculé dynamiquement)
        
    def apply(self, input_grid):
        """
        Applique le pattern de transformation à une grille d'entrée
        
        Args:
            input_grid: Grille d'entrée (numpy array 2D)
            
        Returns:
            output_grid: Grille de sortie transformée
        """
        raise NotImplementedError("Méthode abstraite à implémenter dans les sous-classes")
        
    def matches(self, input_grid, output_grid):
        """
        Vérifie si ce pattern explique la transformation de l'entrée vers la sortie
        
        Args:
            input_grid: Grille d'entrée (numpy array 2D)
            output_grid: Grille de sortie (numpy array 2D)
            
        Returns:
            float: Score entre 0 (ne correspond pas) et 1 (correspond parfaitement)
        """
        try:
            predicted = self.apply(input_grid)
            
            # Si les dimensions ne correspondent pas, score nul
            if predicted.shape != output_grid.shape:
                return 0.0
                
            # Calculer le pourcentage de cellules correctement prédites
            correct_cells = np.sum(predicted == output_grid)
            total_cells = output_grid.size
            
            return correct_cells / total_cells
        except Exception as e:
            logger.warning(f"Erreur lors de l'évaluation du pattern {self.name}: {str(e)}")
            return 0.0
            
    def update_applicability(self, examples):
        """
        Met à jour le score d'applicabilité du pattern en fonction des exemples
        
        Args:
            examples: Liste de tuples (input_grid, output_grid)
            
        Returns:
            float: Score d'applicabilité mis à jour
        """
        scores = []
        for input_grid, output_grid in examples:
            score = self.matches(input_grid, output_grid)
            scores.append(score)
            
        self.applicability_score = np.mean(scores) if scores else 0.0
        return self.applicability_score
        
    def __str__(self):
        return f"Pattern {self.name} (complexité: {self.complexity}, applicabilité: {self.applicability_score:.2f})"


class IdentityPattern(TransformationPatternBase):
    """Pattern qui retourne l'entrée inchangée"""
    
    def __init__(self):
        super().__init__("Identity")
        self.complexity = 0.1  # Pattern le plus simple
        
    def apply(self, input_grid):
        return np.copy(input_grid)


class HorizontalFlipPattern(TransformationPatternBase):
    """Pattern qui retourne l'entrée inversée horizontalement"""
    
    def __init__(self):
        super().__init__("HorizontalFlip")
        self.complexity = 0.3
        
    def apply(self, input_grid):
        return np.fliplr(input_grid)


class VerticalFlipPattern(TransformationPatternBase):
    """Pattern qui retourne l'entrée inversée verticalement"""
    
    def __init__(self):
        super().__init__("VerticalFlip")
        self.complexity = 0.3
        
    def apply(self, input_grid):
        return np.flipud(input_grid)


class RotatePattern(TransformationPatternBase):
    """Pattern qui effectue une rotation de la grille"""
    
    def __init__(self, k=1):
        super().__init__(f"Rotate_{k*90}")
        self.k = k  # Nombre de rotations de 90 degrés
        self.complexity = 0.4
        
    def apply(self, input_grid):
        return np.rot90(input_grid, k=self.k)


class ColorMapPattern(TransformationPatternBase):
    """Pattern qui effectue une transformation de couleur"""
    
    def __init__(self, color_map=None):
        super().__init__("ColorMap")
        self.color_map = color_map or {}  # Dictionnaire de mapping {ancienne_valeur: nouvelle_valeur}
        self.complexity = 0.6
        
    def learn_mapping(self, input_grid, output_grid):
        """Apprend le mapping de couleurs à partir d'un exemple"""
        if input_grid.shape != output_grid.shape:
            return False
            
        # Réinitialiser le mapping
        self.color_map = {}
        
        # Apprendre le mapping
        for i in range(input_grid.shape[0]):
            for j in range(input_grid.shape[1]):
                src_color = input_grid[i, j]
                dst_color = output_grid[i, j]
                
                # Si le mapping est déjà défini et différent, ce pattern ne s'applique pas
                if src_color in self.color_map and self.color_map[src_color] != dst_color:
                    return False
                    
                self.color_map[src_color] = dst_color
                
        return True
        
    def apply(self, input_grid):
        output_grid = np.zeros_like(input_grid)
        
        # Appliquer le mapping de couleurs
        for i in range(input_grid.shape[0]):
            for j in range(input_grid.shape[1]):
                src_color = input_grid[i, j]
                # Si la couleur source est dans le mapping, utiliser la couleur cible
                # Sinon, conserver la couleur source
                output_grid[i, j] = self.color_map.get(src_color, src_color)
                
        return output_grid
        
    def matches(self, input_grid, output_grid):
        """Vérifie si ce pattern explique la transformation"""
        # Si les dimensions sont différentes, ce pattern ne s'applique pas
        if input_grid.shape != output_grid.shape:
            return 0.0
            
        # Essayer d'apprendre le mapping à partir de cet exemple
        if not self.learn_mapping(input_grid, output_grid):
            return 0.0
            
        # Appliquer le mapping et vérifier la correspondance
        return super().matches(input_grid, output_grid)


class CompositePattern(TransformationPatternBase):
    """Pattern composé de plusieurs patterns appliqués séquentiellement"""
    
    def __init__(self, patterns, name=None):
        self.patterns = patterns
        name = name or " -> ".join(p.name for p in patterns)
        super().__init__(name)
        
        # Complexité = somme des complexités des patterns individuels
        self.complexity = sum(p.complexity for p in patterns)
        
    def apply(self, input_grid):
        result = np.copy(input_grid)
        for pattern in self.patterns:
            result = pattern.apply(result)
        return result


class NeuraxTransformationSystem:
    """Système de transformation basé sur Neurax pour les puzzles ARC"""
    
    def __init__(self, grid_size=QUANTUM_GRID_SIZE, time_steps=NUM_TIME_STEPS):
        self.grid_size = grid_size
        self.time_steps = time_steps
        
        # Initialiser le simulateur Neurax
        self.simulator = QuantumGravitySimulator(grid_size=grid_size, time_steps=time_steps)
        
        # Bibliothèque de patterns de transformation
        self.pattern_library = self._initialize_pattern_library()
        
        # Mappings d'encodage/décodage pour représenter les grilles ARC dans l'espace-temps
        self.encoding_mapping = {}
        self.decoding_mapping = {}
        
        # Conservation des meilleurs patterns par puzzle
        self.puzzle_patterns = {}
        
        logger.info(f"Système de transformation Neurax initialisé avec grille {grid_size}³ et {time_steps} pas temporels")
        
    def _initialize_pattern_library(self):
        """Initialise la bibliothèque des patterns de transformation"""
        patterns = [
            IdentityPattern(),
            HorizontalFlipPattern(),
            VerticalFlipPattern(),
            RotatePattern(k=1),  # 90 degrés
            RotatePattern(k=2),  # 180 degrés
            RotatePattern(k=3),  # 270 degrés
            ColorMapPattern()
        ]
        
        # Ajouter quelques patterns composites
        composite_patterns = [
            CompositePattern([HorizontalFlipPattern(), VerticalFlipPattern()]),
            CompositePattern([RotatePattern(k=1), HorizontalFlipPattern()]),
            CompositePattern([ColorMapPattern(), RotatePattern(k=2)])
        ]
        
        patterns.extend(composite_patterns)
        
        logger.info(f"Bibliothèque de patterns initialisée avec {len(patterns)} patterns de base")
        return patterns
        
    def encode_grid_to_spacetime(self, grid, position=(0, 0, 0), time_slice=0):
        """
        Encode une grille ARC dans l'espace-temps du simulateur Neurax
        
        Args:
            grid: Grille ARC (numpy array 2D)
            position: Tuple (x, y, z) indiquant la position dans l'espace
            time_slice: Indice temporel où placer la grille
            
        Returns:
            bool: Succès de l'encodage
        """
        try:
            x, y, z = position
            height, width = grid.shape
            
            # Vérifier que la grille tient dans l'espace-temps
            if (x + width > self.grid_size or 
                y + height > self.grid_size or 
                z + 1 > self.grid_size or
                time_slice >= self.time_steps):
                logger.warning(f"Grille {grid.shape} trop grande pour la position {position} dans l'espace-temps")
                return False
                
            # Encoder la grille dans l'espace-temps
            # Conversion des valeurs ARC (0-9) vers l'espace continu du simulateur
            for i in range(height):
                for j in range(width):
                    arc_value = grid[i, j]
                    
                    # Utiliser un mapping cohérent pour les valeurs
                    if arc_value not in self.encoding_mapping:
                        # Valeurs réparties entre -100 et +100 avec espacement régulier
                        self.encoding_mapping[arc_value] = (arc_value * 20) - 90
                        self.decoding_mapping[self.encoding_mapping[arc_value]] = arc_value
                        
                    # Encoder dans l'espace-temps
                    self.simulator.space_time[time_slice, z, y + i, x + j] = self.encoding_mapping[arc_value]
                    
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'encodage de la grille dans l'espace-temps: {str(e)}")
            return False
            
    def decode_spacetime_to_grid(self, position=(0, 0, 0), size=None, time_slice=0):
        """
        Décode une portion de l'espace-temps en grille ARC
        
        Args:
            position: Tuple (x, y, z) indiquant la position dans l'espace
            size: Tuple (height, width) indiquant la taille de la grille à extraire
            time_slice: Indice temporel d'où extraire la grille
            
        Returns:
            numpy array 2D: Grille ARC décodée
        """
        try:
            x, y, z = position
            height, width = size
            
            # Vérifier que la zone à extraire est valide
            if (x + width > self.grid_size or 
                y + height > self.grid_size or 
                z + 1 > self.grid_size or
                time_slice >= self.time_steps):
                logger.warning(f"Dimensions d'extraction {size} invalides pour la position {position}")
                return None
                
            # Extraire les valeurs de l'espace-temps
            grid_values = self.simulator.space_time[time_slice, z, y:y+height, x:x+width]
            
            # Créer la grille de sortie
            output_grid = np.zeros((height, width), dtype=np.int32)
            
            # Convertir les valeurs continues en valeurs discrètes ARC (0-9)
            for i in range(height):
                for j in range(width):
                    continuous_value = grid_values[i, j]
                    
                    # Trouver la valeur ARC la plus proche dans le mapping
                    nearest_key = min(self.decoding_mapping.keys(), 
                                     key=lambda k: abs(k - continuous_value),
                                     default=0)
                    
                    output_grid[i, j] = self.decoding_mapping.get(nearest_key, 0)
                    
            return output_grid
            
        except Exception as e:
            logger.error(f"Erreur lors du décodage de l'espace-temps en grille ARC: {str(e)}")
            return None
    
    def simulate_transformation(self, input_grid, steps=10, intensity=1.5):
        """
        Utilise le simulateur Neurax pour transformer une grille d'entrée
        
        Args:
            input_grid: Grille d'entrée (numpy array 2D)
            steps: Nombre de pas de simulation
            intensity: Intensité des fluctuations quantiques
            
        Returns:
            numpy array 2D: Grille transformée
        """
        try:
            # Réinitialiser le simulateur
            self.simulator = QuantumGravitySimulator(grid_size=self.grid_size, time_steps=self.time_steps)
            
            # Dimensions de la grille d'entrée
            height, width = input_grid.shape
            
            # Position centrale dans l'espace
            x = (self.grid_size - width) // 2
            y = (self.grid_size - height) // 2
            z = self.grid_size // 2
            
            # Encoder la grille d'entrée dans le premier pas temporel
            self.encode_grid_to_spacetime(input_grid, position=(x, y, z), time_slice=0)
            
            # Appliquer des fluctuations quantiques pour initialiser la transformation
            self.simulator.quantum_fluctuations(intensity=intensity)
            
            # Simuler plusieurs pas pour faire évoluer le système
            for _ in range(steps):
                self.simulator.simulate_step()
                
            # Extraire la grille transformée depuis le dernier pas temporel
            output_grid = self.decode_spacetime_to_grid(
                position=(x, y, z), 
                size=(height, width), 
                time_slice=self.time_steps - 1
            )
            
            return output_grid
            
        except Exception as e:
            logger.error(f"Erreur lors de la simulation de transformation: {str(e)}")
            return None
            
    def train_on_example(self, input_grid, output_grid, max_iterations=100):
        """
        Entraîne le système sur un exemple de transformation (entrée -> sortie)
        
        Args:
            input_grid: Grille d'entrée (numpy array 2D)
            output_grid: Grille de sortie (numpy array 2D)
            max_iterations: Nombre maximal d'itérations d'entraînement
            
        Returns:
            dict: Résultats d'entraînement
        """
        logger.info(f"Entraînement sur exemple: {input_grid.shape} -> {output_grid.shape}")
        
        # Essayer d'abord avec les patterns existants
        best_pattern = None
        best_score = 0.0
        
        # Évaluer tous les patterns de la bibliothèque
        for pattern in self.pattern_library:
            score = pattern.matches(input_grid, output_grid)
            
            if score > best_score:
                best_score = score
                best_pattern = pattern
                
        # Si on a trouvé un pattern qui correspond parfaitement, pas besoin d'entraînement supplémentaire
        if best_score >= 0.99 and best_pattern is not None:
            logger.info(f"Pattern parfait trouvé sans entraînement: {best_pattern.name} (score: {best_score:.4f})")
            return {
                "success": True,
                "iterations": 0,
                "final_score": best_score,
                "pattern": best_pattern.name
            }
            
        # Sinon, essayer d'apprendre avec le simulateur Neurax
        best_neurax_result = None
        best_neurax_score = 0.0
        
        # Paramètres à tester
        intensity_values = [0.5, 1.0, 1.5, 2.0, 3.0]
        steps_values = [5, 10, 15, 20]
        
        total_configs = len(intensity_values) * len(steps_values)
        logger.info(f"Test de {total_configs} configurations du simulateur Neurax")
        
        for intensity in intensity_values:
            for steps in steps_values:
                # Simuler la transformation
                try:
                    result = self.simulate_transformation(
                        input_grid, 
                        steps=steps,
                        intensity=intensity
                    )
                    
                    if result is None:
                        continue
                        
                    # Évaluer le résultat
                    score = self.calculate_accuracy(result, output_grid)
                    
                    if score > best_neurax_score:
                        best_neurax_score = score
                        best_neurax_result = {
                            "result": result,
                            "intensity": intensity,
                            "steps": steps,
                            "score": score
                        }
                except Exception as e:
                    logger.warning(f"Erreur lors du test de configuration (intensity={intensity}, steps={steps}): {str(e)}")
        
        # Déterminer le meilleur résultat global
        if best_neurax_score > best_score and best_neurax_result is not None:
            logger.info(f"Meilleur résultat obtenu avec simulateur Neurax: score={best_neurax_score:.4f}, "
                      f"intensity={best_neurax_result['intensity']}, steps={best_neurax_result['steps']}")
            
            return {
                "success": best_neurax_score >= 0.8,  # Seuil de succès à 80%
                "iterations": best_neurax_result["steps"],
                "final_score": best_neurax_score,
                "pattern": "NeuraxTransformation",
                "parameters": {
                    "intensity": best_neurax_result["intensity"],
                    "steps": best_neurax_result["steps"]
                }
            }
        elif best_pattern is not None:
            logger.info(f"Meilleur résultat obtenu avec pattern classique: {best_pattern.name} (score: {best_score:.4f})")
            
            return {
                "success": best_score >= 0.8,  # Seuil de succès à 80%
                "iterations": 0,
                "final_score": best_score,
                "pattern": best_pattern.name
            }
        else:
            # Aucun pattern ni simulation n'a donné de résultat valide
            logger.warning(f"Aucun pattern ni simulation n'a donné de résultat valide pour cet exemple")
            
            return {
                "success": False,
                "iterations": 0,
                "final_score": 0.0,
                "pattern": "Identity"  # Utiliser l'identité par défaut
            }
            
    def train_on_puzzle(self, puzzle_id, train_pairs, max_iterations=MAX_ITERATIONS):
        """
        Entraîne le système sur un puzzle complet (plusieurs exemples)
        
        Args:
            puzzle_id: Identifiant du puzzle
            train_pairs: Liste de paires (input, output) pour l'entraînement
            max_iterations: Nombre maximal d'itérations d'entraînement
            
        Returns:
            dict: Résultats d'entraînement
        """
        logger.info(f"Entraînement sur puzzle {puzzle_id} avec {len(train_pairs)} exemples")
        
        examples = []
        for pair in train_pairs:
            # Convertir les grilles en arrays numpy
            input_grid = np.array(pair["input"], dtype=np.int32)
            output_grid = np.array(pair["output"], dtype=np.int32)
            
            examples.append((input_grid, output_grid))
            
        # Tester chaque pattern sur tous les exemples
        pattern_scores = {}
        
        for pattern in self.pattern_library:
            pattern.update_applicability(examples)
            pattern_scores[pattern.name] = pattern.applicability_score
            
        # Trouver le meilleur pattern
        best_pattern_name = max(pattern_scores, key=pattern_scores.get)
        best_pattern_score = pattern_scores[best_pattern_name]
        best_pattern = next(p for p in self.pattern_library if p.name == best_pattern_name)
        
        # Si le meilleur pattern est très bon (>90% sur tous les exemples), l'utiliser
        if best_pattern_score >= 0.9:
            logger.info(f"Pattern classique trouvé pour le puzzle {puzzle_id}: {best_pattern_name} "
                       f"(score: {best_pattern_score:.4f})")
            
            # Conserver le pattern pour ce puzzle
            self.puzzle_patterns[puzzle_id] = {
                "pattern_type": "classical",
                "pattern_name": best_pattern_name,
                "score": best_pattern_score,
                "parameters": {}
            }
            
            return {
                "puzzle_id": puzzle_id,
                "success": True,
                "iterations": 0,
                "average_score": best_pattern_score,
                "pattern": best_pattern_name,
                "example_scores": {i: best_pattern.matches(ex[0], ex[1]) for i, ex in enumerate(examples)},
                "training_method": "classical_pattern"
            }
            
        # Sinon, essayer d'apprendre avec le simulateur Neurax
        neurax_results = []
        
        for i, (input_grid, output_grid) in enumerate(examples):
            result = self.train_on_example(input_grid, output_grid, max_iterations=max_iterations//len(examples))
            neurax_results.append(result)
            
        # Analyser les résultats
        success_count = sum(1 for r in neurax_results if r["success"])
        average_score = np.mean([r["final_score"] for r in neurax_results])
        
        # Si tous les exemples sont réussis, c'est un succès global
        if success_count == len(examples):
            logger.info(f"Apprentissage réussi pour le puzzle {puzzle_id} avec score moyen de {average_score:.4f}")
            
            # Déterminer le type de solution
            if all(r["pattern"] != "NeuraxTransformation" for r in neurax_results):
                # Solution par patterns classiques
                pattern_counts = {}
                for r in neurax_results:
                    pattern_counts[r["pattern"]] = pattern_counts.get(r["pattern"], 0) + 1
                
                most_common_pattern = max(pattern_counts, key=pattern_counts.get)
                
                # Conserver le pattern pour ce puzzle
                self.puzzle_patterns[puzzle_id] = {
                    "pattern_type": "classical",
                    "pattern_name": most_common_pattern,
                    "score": average_score,
                    "parameters": {}
                }
                
                return {
                    "puzzle_id": puzzle_id,
                    "success": True,
                    "iterations": sum(r["iterations"] for r in neurax_results),
                    "average_score": average_score,
                    "pattern": most_common_pattern,
                    "example_scores": {i: r["final_score"] for i, r in enumerate(neurax_results)},
                    "training_method": "classical_pattern"
                }
            else:
                # Solution par transformation Neurax
                # Extraire les paramètres optimaux
                neurax_params = [r["parameters"] for r in neurax_results if "parameters" in r]
                
                if neurax_params:
                    # Calculer les paramètres moyens
                    avg_intensity = np.mean([p["intensity"] for p in neurax_params])
                    avg_steps = int(np.mean([p["steps"] for p in neurax_params]))
                    
                    # Conserver les paramètres pour ce puzzle
                    self.puzzle_patterns[puzzle_id] = {
                        "pattern_type": "neurax",
                        "pattern_name": "NeuraxTransformation",
                        "score": average_score,
                        "parameters": {
                            "intensity": avg_intensity,
                            "steps": avg_steps
                        }
                    }
                    
                    return {
                        "puzzle_id": puzzle_id,
                        "success": True,
                        "iterations": sum(r["iterations"] for r in neurax_results),
                        "average_score": average_score,
                        "pattern": "NeuraxTransformation",
                        "parameters": {
                            "intensity": avg_intensity,
                            "steps": avg_steps
                        },
                        "example_scores": {i: r["final_score"] for i, r in enumerate(neurax_results)},
                        "training_method": "neurax_transformation"
                    }
                    
        # Apprentissage partiellement réussi ou échoué
        logger.warning(f"Apprentissage partiel pour le puzzle {puzzle_id}: {success_count}/{len(examples)} "
                     f"exemples réussis, score moyen: {average_score:.4f}")
        
        # Même si l'apprentissage n'est pas parfait, conserver les meilleurs paramètres pour ce puzzle
        if average_score >= 0.5:  # Seuil minimal pour considérer les paramètres comme utilisables
            # Déterminer le type de solution prédominant
            if sum(1 for r in neurax_results if r["pattern"] == "NeuraxTransformation") > len(neurax_results) / 2:
                # Solution principalement par Neurax
                neurax_params = [r["parameters"] for r in neurax_results if "parameters" in r]
                
                if neurax_params:
                    # Calculer les paramètres moyens
                    avg_intensity = np.mean([p["intensity"] for p in neurax_params])
                    avg_steps = int(np.mean([p["steps"] for p in neurax_params]))
                    
                    self.puzzle_patterns[puzzle_id] = {
                        "pattern_type": "neurax",
                        "pattern_name": "NeuraxTransformation",
                        "score": average_score,
                        "parameters": {
                            "intensity": avg_intensity,
                            "steps": avg_steps
                        }
                    }
            else:
                # Solution principalement par patterns classiques
                pattern_counts = {}
                for r in neurax_results:
                    if r["pattern"] != "NeuraxTransformation":
                        pattern_counts[r["pattern"]] = pattern_counts.get(r["pattern"], 0) + 1
                
                if pattern_counts:
                    most_common_pattern = max(pattern_counts, key=pattern_counts.get)
                    
                    self.puzzle_patterns[puzzle_id] = {
                        "pattern_type": "classical",
                        "pattern_name": most_common_pattern,
                        "score": average_score,
                        "parameters": {}
                    }
        
        return {
            "puzzle_id": puzzle_id,
            "success": False,
            "iterations": sum(r["iterations"] for r in neurax_results),
            "average_score": average_score,
            "example_scores": {i: r["final_score"] for i, r in enumerate(neurax_results)},
            "training_method": "mixed"
        }
    
    def predict(self, puzzle_id, test_input):
        """
        Prédit la sortie pour une entrée de test
        
        Args:
            puzzle_id: Identifiant du puzzle
            test_input: Grille d'entrée de test
            
        Returns:
            Grille prédite
        """
        # Convertir l'entrée en array numpy
        input_grid = np.array(test_input, dtype=np.int32)
        
        # Si on a un pattern pour ce puzzle, l'utiliser
        if puzzle_id in self.puzzle_patterns:
            pattern_info = self.puzzle_patterns[puzzle_id]
            
            if pattern_info["pattern_type"] == "classical":
                # Utiliser un pattern classique
                pattern_name = pattern_info["pattern_name"]
                pattern = next((p for p in self.pattern_library if p.name == pattern_name), None)
                
                if pattern:
                    logger.info(f"Prédiction pour puzzle {puzzle_id} avec pattern classique {pattern_name}")
                    return pattern.apply(input_grid)
                    
            elif pattern_info["pattern_type"] == "neurax":
                # Utiliser la transformation Neurax avec les paramètres optimaux
                params = pattern_info["parameters"]
                
                logger.info(f"Prédiction pour puzzle {puzzle_id} avec transformation Neurax "
                           f"(intensity={params['intensity']}, steps={params['steps']})")
                
                return self.simulate_transformation(
                    input_grid,
                    steps=params["steps"],
                    intensity=params["intensity"]
                )
                
        # Si on n'a pas de pattern spécifique, utiliser une approche générique
        logger.warning(f"Pas de pattern spécifique pour le puzzle {puzzle_id}, utilisation de l'approche générique")
        
        # Essayer tous les patterns et prendre le plus probable
        best_score = 0.0
        best_prediction = None
        
        # Essayer les patterns classiques
        for pattern in self.pattern_library:
            try:
                prediction = pattern.apply(input_grid)
                
                # On ne peut pas évaluer directement la qualité de la prédiction sans connaître la sortie attendue
                # On utilise donc un score basé sur des heuristiques
                score = 0.5  # Score par défaut
                
                # Les patterns plus simples sont préférés (principe de parcimonie)
                score += (1.0 - pattern.complexity) * 0.3
                
                # Si ce pattern a bien fonctionné sur d'autres puzzles, le favoriser
                for pid, pinfo in self.puzzle_patterns.items():
                    if pinfo["pattern_type"] == "classical" and pinfo["pattern_name"] == pattern.name:
                        score += pinfo["score"] * 0.2
                
                if score > best_score:
                    best_score = score
                    best_prediction = prediction
            except Exception as e:
                logger.warning(f"Erreur lors de l'application du pattern {pattern.name}: {str(e)}")
                
        # Essayer la transformation Neurax avec différents paramètres
        try:
            # Paramètres génériques qui ont bien fonctionné sur d'autres puzzles
            neurax_score = 0.0
            neurax_prediction = None
            
            # Paramètres à tester
            intensity_values = [1.0, 1.5, 2.0]
            steps_values = [10, 15]
            
            for intensity in intensity_values:
                for steps in steps_values:
                    prediction = self.simulate_transformation(
                        input_grid,
                        steps=steps,
                        intensity=intensity
                    )
                    
                    if prediction is None:
                        continue
                    
                    # Score basé sur des heuristiques
                    score = 0.4  # Score par défaut
                    
                    # Si ces paramètres ont bien fonctionné sur d'autres puzzles, les favoriser
                    for pid, pinfo in self.puzzle_patterns.items():
                        if pinfo["pattern_type"] == "neurax":
                            # Similarité des paramètres
                            i_sim = 1.0 - min(abs(pinfo["parameters"]["intensity"] - intensity) / 3.0, 1.0)
                            s_sim = 1.0 - min(abs(pinfo["parameters"]["steps"] - steps) / 20.0, 1.0)
                            param_sim = (i_sim + s_sim) / 2.0
                            
                            score += pinfo["score"] * param_sim * 0.3
                    
                    if score > neurax_score:
                        neurax_score = score
                        neurax_prediction = prediction
                        
            # Comparer le meilleur résultat Neurax avec le meilleur pattern classique
            if neurax_score > best_score and neurax_prediction is not None:
                best_prediction = neurax_prediction
                
        except Exception as e:
            logger.warning(f"Erreur lors de la transformation Neurax générique: {str(e)}")
            
        # Si aucune prédiction n'a été générée, retourner une grille identique
        if best_prediction is None:
            logger.warning(f"Aucune prédiction valide pour le puzzle {puzzle_id}, retour de l'identité")
            best_prediction = np.copy(input_grid)
            
        return best_prediction
        
    def calculate_accuracy(self, predicted_grid, expected_grid):
        """
        Calcule la précision d'une prédiction
        
        Args:
            predicted_grid: Grille prédite
            expected_grid: Grille attendue
            
        Returns:
            float: Précision entre 0 et 1
        """
        # Si les dimensions ne correspondent pas, précision nulle
        if predicted_grid.shape != expected_grid.shape:
            return 0.0
            
        # Pourcentage de cellules correctes
        correct_cells = np.sum(predicted_grid == expected_grid)
        total_cells = expected_grid.size
        
        return correct_cells / total_cells


class ARCPuzzleProcessor:
    """Classe pour le traitement des puzzles ARC avec le système Neurax"""
    
    def __init__(self, data_path="../arc_data"):
        self.data_path = data_path
        self.arc_data = self._load_arc_data()
        
        # Initialiser le système de transformation Neurax
        self.transformation_system = NeuraxTransformationSystem()
        
        # Résultats d'apprentissage et de prédiction
        self.training_results = {}
        self.evaluation_results = {}
        self.test_results = {}
        
        logger.info("Processeur de puzzles ARC initialisé")
        
    def _load_arc_data(self):
        """Charge les données des puzzles ARC"""
        try:
            # Chargement des fichiers de puzzle
            training_path = os.path.join(self.data_path, "arc-agi_training_challenges.json")
            evaluation_path = os.path.join(self.data_path, "arc-agi_evaluation_challenges.json")
            test_path = os.path.join(self.data_path, "arc-agi_test_challenges.json")
            
            # Solutions pour évaluation
            training_solutions_path = os.path.join(self.data_path, "arc-agi_training_solutions.json")
            evaluation_solutions_path = os.path.join(self.data_path, "arc-agi_evaluation_solutions.json")
            
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
            return None
            
    def train_on_all_training_puzzles(self, limit=None):
        """
        Entraîne le système sur tous les puzzles d'entraînement
        
        Args:
            limit: Nombre maximal de puzzles à traiter (None = tous)
            
        Returns:
            dict: Résultats d'entraînement
        """
        if not self.arc_data:
            logger.error("Données ARC non disponibles")
            return {}
            
        # Récupérer les puzzles d'entraînement
        puzzles = self.arc_data["training"]["challenges"]
        
        # Limiter le nombre de puzzles si demandé
        puzzle_ids = list(puzzles.keys())
        if limit:
            puzzle_ids = puzzle_ids[:limit]
            
        logger.info(f"Démarrage de l'entraînement sur {len(puzzle_ids)} puzzles")
        
        # Entraîner sur chaque puzzle
        results = {}
        
        for idx, puzzle_id in enumerate(tqdm(puzzle_ids, desc="Entraînement")):
            puzzle = puzzles[puzzle_id]
            train_pairs = puzzle["train"]
            
            # Entraîner le système sur ce puzzle
            result = self.transformation_system.train_on_puzzle(puzzle_id, train_pairs)
            
            # Conserver le résultat
            results[puzzle_id] = result
            
            # Log périodique
            if (idx + 1) % 10 == 0 or idx == 0 or idx == len(puzzle_ids) - 1:
                # Calculer les statistiques intermédiaires
                success_count = sum(1 for r in results.values() if r["success"])
                avg_score = np.mean([r["average_score"] for r in results.values()])
                
                logger.info(f"Progression: {idx+1}/{len(puzzle_ids)} puzzles, "
                           f"Taux de réussite: {success_count/(idx+1)*100:.2f}%, "
                           f"Score moyen: {avg_score:.4f}")
                
        # Calculer les statistiques finales
        success_count = sum(1 for r in results.values() if r["success"])
        success_rate = success_count / len(results) if results else 0
        avg_score = np.mean([r["average_score"] for r in results.values()]) if results else 0
        
        logger.info(f"Entraînement terminé sur {len(results)} puzzles: "
                   f"Taux de réussite: {success_rate*100:.2f}%, "
                   f"Score moyen: {avg_score:.4f}")
                   
        # Conserver les résultats
        self.training_results = results
        
        return {
            "puzzle_count": len(results),
            "success_count": success_count,
            "success_rate": success_rate,
            "average_score": avg_score,
            "details": results
        }
        
    def evaluate_on_evaluation_puzzles(self, limit=None):
        """
        Évalue le système sur les puzzles d'évaluation
        
        Args:
            limit: Nombre maximal de puzzles à traiter (None = tous)
            
        Returns:
            dict: Résultats d'évaluation
        """
        if not self.arc_data:
            logger.error("Données ARC non disponibles")
            return {}
            
        # Récupérer les puzzles d'évaluation
        puzzles = self.arc_data["evaluation"]["challenges"]
        solutions = self.arc_data["evaluation"]["solutions"]
        
        # Limiter le nombre de puzzles si demandé
        puzzle_ids = list(puzzles.keys())
        if limit:
            puzzle_ids = puzzle_ids[:limit]
            
        logger.info(f"Démarrage de l'évaluation sur {len(puzzle_ids)} puzzles")
        
        # Évaluer sur chaque puzzle
        results = {}
        
        for idx, puzzle_id in enumerate(tqdm(puzzle_ids, desc="Évaluation")):
            puzzle = puzzles[puzzle_id]
            
            # Générer des prédictions pour chaque entrée de test
            puzzle_results = []
            
            for i, test_input in enumerate(puzzle["test"]):
                # Récupérer l'entrée de test
                test_input_grid = np.array(test_input["input"], dtype=np.int32)
                
                # Générer la prédiction
                prediction = self.transformation_system.predict(puzzle_id, test_input_grid)
                
                # Convertir la prédiction en liste pour la compatibilité JSON
                prediction_list = prediction.tolist() if prediction is not None else None
                
                # Récupérer la solution attendue si disponible
                expected_output = None
                accuracy = None
                
                if solutions and puzzle_id in solutions and i < len(solutions[puzzle_id]):
                    expected_output = solutions[puzzle_id][i]["output"]
                    expected_output_grid = np.array(expected_output, dtype=np.int32)
                    
                    # Calculer la précision
                    accuracy = self.transformation_system.calculate_accuracy(prediction, expected_output_grid)
                
                # Conserver le résultat
                puzzle_results.append({
                    "test_idx": i,
                    "prediction": prediction_list,
                    "expected_output": expected_output,
                    "accuracy": accuracy
                })
                
            # Calculer la précision moyenne pour ce puzzle
            accuracies = [r["accuracy"] for r in puzzle_results if r["accuracy"] is not None]
            avg_accuracy = np.mean(accuracies) if accuracies else None
            
            # Conserver le résultat global pour ce puzzle
            results[puzzle_id] = {
                "test_count": len(puzzle_results),
                "average_accuracy": avg_accuracy,
                "details": puzzle_results
            }
            
            # Log périodique
            if (idx + 1) % 10 == 0 or idx == 0 or idx == len(puzzle_ids) - 1:
                # Calculer les statistiques intermédiaires
                valid_results = [r for r in results.values() if r["average_accuracy"] is not None]
                avg_acc = np.mean([r["average_accuracy"] for r in valid_results]) if valid_results else 0
                
                logger.info(f"Progression: {idx+1}/{len(puzzle_ids)} puzzles, "
                           f"Précision moyenne: {avg_acc*100:.2f}%")
                
        # Calculer les statistiques finales
        valid_results = [r for r in results.values() if r["average_accuracy"] is not None]
        avg_accuracy = np.mean([r["average_accuracy"] for r in valid_results]) if valid_results else 0
        
        # Calculer le taux de réussite parfaite (100% d'exactitude)
        perfect_count = sum(1 for r in valid_results if r["average_accuracy"] == 1.0)
        perfect_rate = perfect_count / len(valid_results) if valid_results else 0
        
        logger.info(f"Évaluation terminée sur {len(results)} puzzles: "
                   f"Précision moyenne: {avg_accuracy*100:.2f}%, "
                   f"Taux de réussite parfaite: {perfect_rate*100:.2f}%")
                   
        # Conserver les résultats
        self.evaluation_results = results
        
        return {
            "puzzle_count": len(results),
            "valid_puzzle_count": len(valid_results),
            "average_accuracy": avg_accuracy,
            "perfect_count": perfect_count,
            "perfect_rate": perfect_rate,
            "details": results
        }
        
    def generate_test_predictions(self, limit=None, output_file=None):
        """
        Génère des prédictions pour les puzzles de test
        
        Args:
            limit: Nombre maximal de puzzles à traiter (None = tous)
            output_file: Chemin du fichier de sortie pour les prédictions
            
        Returns:
            dict: Prédictions générées
        """
        if not self.arc_data:
            logger.error("Données ARC non disponibles")
            return {}
            
        # Récupérer les puzzles de test
        puzzles = self.arc_data["test"]["challenges"]
        
        # Limiter le nombre de puzzles si demandé
        puzzle_ids = list(puzzles.keys())
        if limit:
            puzzle_ids = puzzle_ids[:limit]
            
        logger.info(f"Génération de prédictions pour {len(puzzle_ids)} puzzles de test")
        
        # Format de sortie Kaggle
        submission_format = {}
        
        # Générer des prédictions pour chaque puzzle
        results = {}
        
        for idx, puzzle_id in enumerate(tqdm(puzzle_ids, desc="Test")):
            puzzle = puzzles[puzzle_id]
            
            # Initialiser le format de soumission pour ce puzzle
            submission_format[puzzle_id] = []
            
            # Générer des prédictions pour chaque entrée de test
            puzzle_predictions = []
            
            for i, test_input in enumerate(puzzle["test"]):
                # Récupérer l'entrée de test
                test_input_grid = np.array(test_input["input"], dtype=np.int32)
                
                # Générer deux prédictions différentes
                predictions = []
                
                # Première prédiction - paramètres standards
                prediction1 = self.transformation_system.predict(puzzle_id, test_input_grid)
                predictions.append(prediction1.tolist() if prediction1 is not None else [[0, 0], [0, 0]])
                
                # Seconde prédiction - paramètres alternatifs
                # Si on n'a pas de pattern spécifique pour ce puzzle, utiliser des paramètres génériques
                if puzzle_id in self.transformation_system.puzzle_patterns:
                    pattern_info = self.transformation_system.puzzle_patterns[puzzle_id]
                    
                    if pattern_info["pattern_type"] == "neurax":
                        # Modifier légèrement les paramètres
                        intensity = pattern_info["parameters"]["intensity"] * 1.2  # 20% plus fort
                        steps = pattern_info["parameters"]["steps"] + 5  # 5 pas de plus
                        
                        prediction2 = self.transformation_system.simulate_transformation(
                            test_input_grid,
                            steps=steps,
                            intensity=intensity
                        )
                    else:
                        # Utiliser un pattern différent
                        pattern_name = pattern_info["pattern_name"]
                        current_pattern_idx = next((i for i, p in enumerate(self.transformation_system.pattern_library) 
                                                  if p.name == pattern_name), 0)
                        
                        # Choisir un pattern différent
                        alt_pattern_idx = (current_pattern_idx + 1) % len(self.transformation_system.pattern_library)
                        alt_pattern = self.transformation_system.pattern_library[alt_pattern_idx]
                        
                        prediction2 = alt_pattern.apply(test_input_grid)
                else:
                    # Utiliser des paramètres génériques différents
                    prediction2 = self.transformation_system.simulate_transformation(
                        test_input_grid,
                        steps=15,
                        intensity=2.0
                    )
                
                predictions.append(prediction2.tolist() if prediction2 is not None else [[0, 0], [0, 0]])
                
                # Ajouter au format de soumission
                submission_format[puzzle_id].append({
                    "attempt_1": predictions[0],
                    "attempt_2": predictions[1]
                })
                
                # Conserver les prédictions
                puzzle_predictions.append({
                    "test_idx": i,
                    "prediction1": predictions[0],
                    "prediction2": predictions[1]
                })
                
            # Conserver le résultat global pour ce puzzle
            results[puzzle_id] = {
                "test_count": len(puzzle_predictions),
                "details": puzzle_predictions
            }
            
            # Log périodique
            if (idx + 1) % 10 == 0 or idx == 0 or idx == len(puzzle_ids) - 1:
                logger.info(f"Progression: {idx+1}/{len(puzzle_ids)} puzzles")
                
        # Sauvegarder le fichier de soumission si demandé
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(submission_format, f)
                
            logger.info(f"Prédictions de test sauvegardées dans {output_file}")
            
        # Conserver les résultats
        self.test_results = results
        
        return {
            "puzzle_count": len(results),
            "submission_format": submission_format,
            "details": results
        }
        
    def save_results(self, output_dir):
        """
        Sauvegarde tous les résultats dans des fichiers JSON
        
        Args:
            output_dir: Répertoire de sortie
            
        Returns:
            bool: Succès de la sauvegarde
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Sauvegarder les résultats d'entraînement
            if self.training_results:
                training_path = os.path.join(output_dir, "training_results.json")
                with open(training_path, 'w') as f:
                    json.dump(self.training_results, f, indent=2)
                    
                logger.info(f"Résultats d'entraînement sauvegardés dans {training_path}")
                
            # Sauvegarder les résultats d'évaluation
            if self.evaluation_results:
                evaluation_path = os.path.join(output_dir, "evaluation_results.json")
                with open(evaluation_path, 'w') as f:
                    json.dump(self.evaluation_results, f, indent=2)
                    
                logger.info(f"Résultats d'évaluation sauvegardés dans {evaluation_path}")
                
            # Sauvegarder les résultats de test
            if self.test_results:
                test_path = os.path.join(output_dir, "test_results.json")
                with open(test_path, 'w') as f:
                    json.dump(self.test_results, f, indent=2)
                    
                logger.info(f"Résultats de test sauvegardés dans {test_path}")
                
            # Sauvegarder les patterns appris
            if self.transformation_system.puzzle_patterns:
                patterns_path = os.path.join(output_dir, "learned_patterns.json")
                with open(patterns_path, 'w') as f:
                    json.dump(self.transformation_system.puzzle_patterns, f, indent=2)
                    
                logger.info(f"Patterns appris sauvegardés dans {patterns_path}")
                
            # Sauvegarder un fichier de soumission
            submission_path = os.path.join(output_dir, "submission.json")
            self.generate_test_predictions(output_file=submission_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des résultats: {str(e)}")
            return False
            
    def generate_summary_report(self, output_file=None):
        """
        Génère un rapport de synthèse au format Markdown
        
        Args:
            output_file: Chemin du fichier de sortie
            
        Returns:
            str: Contenu du rapport
        """
        # Calculer les statistiques d'entraînement
        training_stats = {
            "puzzle_count": 0,
            "success_count": 0,
            "success_rate": 0,
            "average_score": 0
        }
        
        if self.training_results:
            results = self.training_results
            training_stats["puzzle_count"] = len(results)
            training_stats["success_count"] = sum(1 for r in results.values() if r.get("success", False))
            training_stats["success_rate"] = training_stats["success_count"] / training_stats["puzzle_count"] if training_stats["puzzle_count"] > 0 else 0
            training_stats["average_score"] = np.mean([r.get("average_score", 0) for r in results.values()]) if results else 0
            
        # Calculer les statistiques d'évaluation
        evaluation_stats = {
            "puzzle_count": 0,
            "valid_puzzle_count": 0,
            "average_accuracy": 0,
            "perfect_count": 0,
            "perfect_rate": 0
        }
        
        if self.evaluation_results:
            results = self.evaluation_results
            evaluation_stats["puzzle_count"] = len(results)
            
            valid_results = [r for r in results.values() if r.get("average_accuracy") is not None]
            evaluation_stats["valid_puzzle_count"] = len(valid_results)
            evaluation_stats["average_accuracy"] = np.mean([r.get("average_accuracy", 0) for r in valid_results]) if valid_results else 0
            
            evaluation_stats["perfect_count"] = sum(1 for r in valid_results if r.get("average_accuracy") == 1.0)
            evaluation_stats["perfect_rate"] = evaluation_stats["perfect_count"] / evaluation_stats["valid_puzzle_count"] if evaluation_stats["valid_puzzle_count"] > 0 else 0
            
        # Calculer les statistiques de test
        test_stats = {
            "puzzle_count": 0
        }
        
        if self.test_results:
            test_stats["puzzle_count"] = len(self.test_results)
            
        # Générer le rapport Markdown
        report = []
        report.append("# Rapport d'Apprentissage et d'Évaluation sur les Puzzles ARC")
        report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Section d'entraînement
        report.append("## Résultats d'Entraînement")
        report.append("")
        report.append(f"- **Nombre de puzzles traités**: {training_stats['puzzle_count']}")
        report.append(f"- **Puzzles appris avec succès**: {training_stats['success_count']} ({training_stats['success_rate']*100:.2f}%)")
        report.append(f"- **Score moyen**: {training_stats['average_score']*100:.2f}%")
        report.append("")
        
        # Distribution des méthodes d'apprentissage
        if self.training_results:
            method_counts = {}
            for puzzle_id, result in self.training_results.items():
                method = result.get("training_method", "unknown")
                method_counts[method] = method_counts.get(method, 0) + 1
                
            report.append("### Distribution des Méthodes d'Apprentissage")
            report.append("")
            report.append("| Méthode | Nombre de Puzzles | Pourcentage |")
            report.append("|---------|------------------|-------------|")
            
            for method, count in method_counts.items():
                percentage = count / training_stats["puzzle_count"] * 100 if training_stats["puzzle_count"] > 0 else 0
                report.append(f"| {method} | {count} | {percentage:.2f}% |")
                
            report.append("")
            
        # Section d'évaluation
        report.append("## Résultats d'Évaluation")
        report.append("")
        report.append(f"- **Nombre de puzzles évalués**: {evaluation_stats['puzzle_count']}")
        report.append(f"- **Précision moyenne**: {evaluation_stats['average_accuracy']*100:.2f}%")
        report.append(f"- **Puzzles résolus parfaitement**: {evaluation_stats['perfect_count']} ({evaluation_stats['perfect_rate']*100:.2f}%)")
        report.append("")
        
        # Histogramme des précisions
        if self.evaluation_results:
            accuracy_ranges = {
                "0-10%": 0,
                "10-20%": 0,
                "20-30%": 0,
                "30-40%": 0,
                "40-50%": 0,
                "50-60%": 0,
                "60-70%": 0,
                "70-80%": 0,
                "80-90%": 0,
                "90-99%": 0,
                "100%": 0
            }
            
            for puzzle_id, result in self.evaluation_results.items():
                accuracy = result.get("average_accuracy")
                if accuracy is not None:
                    if accuracy == 1.0:
                        accuracy_ranges["100%"] += 1
                    else:
                        range_idx = min(int(accuracy * 10), 9)
                        range_key = list(accuracy_ranges.keys())[range_idx]
                        accuracy_ranges[range_key] += 1
                        
            report.append("### Distribution des Précisions")
            report.append("")
            report.append("| Plage de Précision | Nombre de Puzzles | Pourcentage |")
            report.append("|---------------------|------------------|-------------|")
            
            for range_key, count in accuracy_ranges.items():
                percentage = count / evaluation_stats["valid_puzzle_count"] * 100 if evaluation_stats["valid_puzzle_count"] > 0 else 0
                report.append(f"| {range_key} | {count} | {percentage:.2f}% |")
                
            report.append("")
            
        # Section de test
        report.append("## Prédictions de Test")
        report.append("")
        report.append(f"- **Nombre de puzzles de test**: {test_stats['puzzle_count']}")
        report.append("")
        
        # Conclusion
        report.append("## Conclusion")
        report.append("")
        
        if training_stats["success_rate"] >= 0.9 and evaluation_stats["average_accuracy"] >= 0.9:
            conclusion = "Le système a démontré d'excellentes performances, apprenant avec succès la grande majorité des puzzles d'entraînement et généralisant efficacement aux puzzles d'évaluation."
        elif training_stats["success_rate"] >= 0.7 and evaluation_stats["average_accuracy"] >= 0.7:
            conclusion = "Le système a démontré de bonnes performances, avec un taux d'apprentissage élevé et une bonne capacité de généralisation."
        elif training_stats["success_rate"] >= 0.5 and evaluation_stats["average_accuracy"] >= 0.5:
            conclusion = "Le système a démontré des performances moyennes, apprenant environ la moitié des puzzles d'entraînement et généralisant partiellement aux puzzles d'évaluation."
        else:
            conclusion = "Le système a démontré des performances limitées, avec des difficultés à apprendre les puzzles d'entraînement et à généraliser aux puzzles d'évaluation."
            
        report.append(conclusion)
        report.append("")
        
        # Ajouter des recommandations
        report.append("### Recommandations")
        report.append("")
        report.append("1. **Amélioration des patterns de transformation**: Développer une bibliothèque plus complète de patterns spécifiques aux puzzles ARC.")
        report.append("2. **Optimisation des paramètres Neurax**: Affiner les configurations optimales pour différentes catégories de puzzles.")
        report.append("3. **Apprentissage hiérarchique**: Implémenter un système de composition de patterns pour résoudre des puzzles plus complexes.")
        report.append("4. **Intégration avec d'autres approches**: Combiner la simulation quantique avec des techniques d'apprentissage symbolique.")
        report.append("")
        
        # Assembler le rapport complet
        report_text = "\n".join(report)
        
        # Sauvegarder le rapport si demandé
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
                
            logger.info(f"Rapport de synthèse sauvegardé dans {output_file}")
            
        return report_text


def run_complete_arc_learning():
    """
    Exécute l'apprentissage et l'évaluation complète sur tous les puzzles ARC
    """
    logger.info("Démarrage de l'apprentissage et de l'évaluation sur tous les puzzles ARC")
    
    # Initialiser le processeur de puzzles
    processor = ARCPuzzleProcessor()
    
    # Créer le répertoire de sortie
    output_dir = "arc_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Étape 1: Apprentissage sur tous les puzzles d'entraînement
    logger.info("ÉTAPE 1: Apprentissage sur tous les puzzles d'entraînement")
    training_results = processor.train_on_all_training_puzzles()
    
    # Étape 2: Évaluation sur les puzzles d'évaluation
    logger.info("ÉTAPE 2: Évaluation sur les puzzles d'évaluation")
    evaluation_results = processor.evaluate_on_evaluation_puzzles()
    
    # Étape 3: Génération des prédictions pour les puzzles de test
    logger.info("ÉTAPE 3: Génération des prédictions pour les puzzles de test")
    test_results = processor.generate_test_predictions()
    
    # Étape 4: Sauvegarde des résultats
    logger.info("ÉTAPE 4: Sauvegarde des résultats")
    processor.save_results(output_dir)
    
    # Étape 5: Génération du rapport de synthèse
    logger.info("ÉTAPE 5: Génération du rapport de synthèse")
    report_path = os.path.join(output_dir, "summary_report.md")
    processor.generate_summary_report(output_file=report_path)
    
    logger.info(f"Apprentissage et évaluation terminés. Résultats dans {output_dir}")
    
    return {
        "training_results": training_results,
        "evaluation_results": evaluation_results,
        "test_results": test_results,
        "output_dir": output_dir
    }


if __name__ == "__main__":
    run_complete_arc_learning()