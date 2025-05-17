#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de benchmark comparatif entre le simulateur original et le simulateur optimisé
"""

import time
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple

# Configurer le logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Benchmark")

# Importer les simulateurs
from quantum_gravity_sim import QuantumGravitySimulator
from quantum_gravity_sim_gpu import QuantumGravitySimulatorGPU

def run_benchmark(grid_sizes: List[int], time_steps: int = 8, num_puzzles: int = 10) -> Dict[str, Any]:
    """
    Exécute un benchmark complet des simulateurs
    
    Args:
        grid_sizes: Liste des tailles de grille à tester
        time_steps: Nombre d'étapes temporelles
        num_puzzles: Nombre de puzzles à simuler
        
    Returns:
        Résultats du benchmark
    """
    results = {
        "original": {},
        "optimized_cpu": {},
        "optimized_gpu": {},
        "speedups": {}
    }
    
    # Générer des données de test pour les puzzles
    test_puzzles = generate_test_puzzles(num_puzzles, max_grid_size=max(grid_sizes))
    
    # Tester chaque taille de grille
    for grid_size in grid_sizes:
        logger.info(f"=== Benchmark pour grille {grid_size}x{grid_size} ===")
        
        # Simulateur original
        original_times = []
        sim_original = QuantumGravitySimulator(grid_size=grid_size, time_steps=time_steps)
        
        # Simulateur optimisé (CPU)
        optimized_cpu_times = []
        sim_cpu = QuantumGravitySimulatorGPU(grid_size=grid_size, time_steps=time_steps, use_gpu=False)
        
        # Simulateur optimisé (GPU/simulé)
        optimized_gpu_times = []
        sim_gpu = QuantumGravitySimulatorGPU(grid_size=grid_size, time_steps=time_steps, use_gpu=True)
        
        # Tester sur chaque puzzle
        for i, puzzle in enumerate(test_puzzles):
            # Adapter la taille du puzzle à la grille courante
            adapted_puzzle = adapt_puzzle_to_grid(puzzle, grid_size)
            
            # Simulateur original
            start_time = time.time()
            _ = process_with_original(sim_original, adapted_puzzle)
            original_time = time.time() - start_time
            original_times.append(original_time)
            
            # Simulateur optimisé (CPU)
            start_time = time.time()
            _ = sim_cpu.process_puzzle(adapted_puzzle)
            optimized_cpu_time = time.time() - start_time
            optimized_cpu_times.append(optimized_cpu_time)
            
            # Simulateur optimisé (GPU/simulé)
            start_time = time.time()
            _ = sim_gpu.process_puzzle(adapted_puzzle)
            optimized_gpu_time = time.time() - start_time
            optimized_gpu_times.append(optimized_gpu_time)
            
            logger.info(f"Puzzle {i+1}/{num_puzzles}: Original={original_time:.4f}s, CPU={optimized_cpu_time:.4f}s, GPU={optimized_gpu_time:.4f}s")
        
        # Calculer les moyennes
        avg_original = sum(original_times) / len(original_times)
        avg_cpu = sum(optimized_cpu_times) / len(optimized_cpu_times)
        avg_gpu = sum(optimized_gpu_times) / len(optimized_gpu_times)
        
        # Calculer les speedups
        cpu_speedup = avg_original / avg_cpu
        gpu_speedup = avg_original / avg_gpu
        
        # Enregistrer les résultats
        results["original"][grid_size] = avg_original
        results["optimized_cpu"][grid_size] = avg_cpu
        results["optimized_gpu"][grid_size] = avg_gpu
        results["speedups"][grid_size] = {
            "cpu_vs_original": cpu_speedup,
            "gpu_vs_original": gpu_speedup,
            "gpu_vs_cpu": avg_cpu / avg_gpu
        }
        
        logger.info(f"Résultats pour grille {grid_size}x{grid_size}:")
        logger.info(f"- Temps moyen original: {avg_original:.4f}s")
        logger.info(f"- Temps moyen CPU optimisé: {avg_cpu:.4f}s (speedup: {cpu_speedup:.1f}x)")
        logger.info(f"- Temps moyen GPU optimisé: {avg_gpu:.4f}s (speedup: {gpu_speedup:.1f}x)")
    
    return results

def generate_test_puzzles(num_puzzles: int, max_grid_size: int) -> List[Dict[str, Any]]:
    """
    Génère des puzzles de test avec différentes complexités
    
    Args:
        num_puzzles: Nombre de puzzles à générer
        max_grid_size: Taille maximale de la grille
        
    Returns:
        Liste de puzzles de test
    """
    puzzles = []
    
    for i in range(num_puzzles):
        # Varier la taille du puzzle
        size = np.random.randint(3, max(3, max_grid_size // 2))
        
        # Créer une grille aléatoire
        input_grid = np.random.randint(0, 10, size=(size, size)).tolist()
        output_grid = np.random.randint(0, 10, size=(size, size)).tolist()
        
        # Créer un puzzle avec plusieurs exemples (nombre variable)
        num_examples = np.random.randint(1, 4)
        examples = []
        for _ in range(num_examples):
            examples.append({
                "input": input_grid,
                "output": output_grid
            })
        
        puzzle = {
            "id": f"test_puzzle_{i}",
            "train": examples
        }
        
        puzzles.append(puzzle)
    
    return puzzles

def adapt_puzzle_to_grid(puzzle: Dict[str, Any], grid_size: int) -> Dict[str, Any]:
    """
    Adapte un puzzle à une taille de grille spécifique
    
    Args:
        puzzle: Puzzle à adapter
        grid_size: Taille de grille cible
        
    Returns:
        Puzzle adapté
    """
    # Copie pour ne pas modifier l'original
    adapted = puzzle.copy()
    
    # Si pas besoin d'adaptation, retourner tel quel
    return adapted

def process_with_original(simulator: QuantumGravitySimulator, puzzle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Traite un puzzle avec le simulateur original
    
    Args:
        simulator: Instance du simulateur original
        puzzle: Puzzle à traiter
        
    Returns:
        Résultats du traitement
    """
    # Simuler le comportement de process_puzzle pour le simulateur original
    simulator.quantum_fluctuations(1.5)
    simulator.simulate_step()
    
    return {
        "grid_size": simulator.grid_size,
        "time_steps": simulator.time_steps
    }

def plot_results(results: Dict[str, Any], output_file: str = "benchmark_results.png"):
    """
    Génère un graphique des résultats de benchmark
    
    Args:
        results: Résultats du benchmark
        output_file: Fichier de sortie pour le graphique
    """
    grid_sizes = sorted(list(results["original"].keys()))
    
    # Temps d'exécution
    plt.figure(figsize=(12, 10))
    
    # Subplot 1: Temps d'exécution
    plt.subplot(2, 1, 1)
    plt.plot(grid_sizes, [results["original"][s] for s in grid_sizes], 'o-', label='Original')
    plt.plot(grid_sizes, [results["optimized_cpu"][s] for s in grid_sizes], 's-', label='Optimisé (CPU)')
    plt.plot(grid_sizes, [results["optimized_gpu"][s] for s in grid_sizes], '^-', label='Optimisé (GPU)')
    
    plt.xlabel('Taille de la grille')
    plt.ylabel("Temps d'exécution (s)")
    plt.title("Comparaison des temps d'exécution")
    plt.grid(True)
    plt.legend()
    plt.yscale('log')  # Échelle logarithmique pour mieux voir les différences
    
    # Subplot 2: Speedups
    plt.subplot(2, 1, 2)
    plt.plot(grid_sizes, [results["speedups"][s]["cpu_vs_original"] for s in grid_sizes], 's-', label='CPU vs Original')
    plt.plot(grid_sizes, [results["speedups"][s]["gpu_vs_original"] for s in grid_sizes], '^-', label='GPU vs Original')
    plt.plot(grid_sizes, [results["speedups"][s]["gpu_vs_cpu"] for s in grid_sizes], 'v-', label='GPU vs CPU')
    
    plt.xlabel('Taille de la grille')
    plt.ylabel('Accélération (x)')
    plt.title("Facteurs d'accélération")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_file)
    logger.info(f"Graphique sauvegardé: {output_file}")

def main():
    """
    Fonction principale
    """
    logger.info("=== DÉMARRAGE DU BENCHMARK COMPARATIF DES SIMULATEURS ===")
    
    # Définir les tailles de grille à tester
    grid_sizes = [16, 32, 64, 128]
    
    # Exécuter le benchmark
    results = run_benchmark(grid_sizes, time_steps=8, num_puzzles=5)
    
    # Sauvegarder les résultats au format JSON
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Résultats sauvegardés dans benchmark_results.json")
    
    # Générer le graphique
    try:
        plot_results(results)
    except Exception as e:
        logger.error(f"Erreur lors de la génération du graphique: {str(e)}")
    
    # Afficher le résumé
    logger.info("\n=== RÉSUMÉ DES RÉSULTATS ===")
    
    for grid_size in grid_sizes:
        logger.info(f"Grille {grid_size}x{grid_size}:")
        logger.info(f"- Original: {results['original'][grid_size]:.4f}s")
        logger.info(f"- CPU optimisé: {results['optimized_cpu'][grid_size]:.4f}s (speedup: {results['speedups'][grid_size]['cpu_vs_original']:.1f}x)")
        logger.info(f"- GPU optimisé: {results['optimized_gpu'][grid_size]:.4f}s (speedup: {results['speedups'][grid_size]['gpu_vs_original']:.1f}x)")
    
    logger.info("\n=== FIN DU BENCHMARK ===")

if __name__ == "__main__":
    main()