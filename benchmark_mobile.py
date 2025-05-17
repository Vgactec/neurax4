#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Benchmark des performances du simulateur Neurax2 pour appareils mobiles
Compare les différentes configurations en termes de mémoire et vitesse d'exécution
"""

import os
import sys
import time
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("benchmark_mobile.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("BenchmarkMobile")

# Importer les simulateurs
from quantum_gravity_sim_mobile import QuantumGravitySimulatorMobile

def run_benchmark(grid_sizes=[8, 16, 32], time_steps=4, precisions=["float32", "float16", "int8"],
                memory_limits=[50, 100, 200], num_runs=3):
    """
    Exécute un benchmark complet du simulateur mobile
    
    Args:
        grid_sizes: Liste des tailles de grille à tester
        time_steps: Nombre d'étapes temporelles
        precisions: Liste des précisions à tester
        memory_limits: Liste des limites mémoire à tester (MB)
        num_runs: Nombre d'exécutions par configuration pour moyenner
        
    Returns:
        Résultats du benchmark
    """
    results = {
        "configurations": [],
        "performance": [],
        "memory": [],
        "grid_sizes": grid_sizes,
        "time_steps": time_steps,
        "precisions": precisions,
        "memory_limits": memory_limits
    }
    
    total_configs = len(grid_sizes) * len(precisions) * len(memory_limits)
    config_count = 0
    
    for grid_size in grid_sizes:
        for precision in precisions:
            for memory_limit in memory_limits:
                config_count += 1
                logger.info(f"Configuration {config_count}/{total_configs}: "
                          f"Grille {grid_size}x{grid_size}, Précision {precision}, "
                          f"Limite mémoire {memory_limit} MB")
                
                # Vérifier si la configuration est possible
                try:
                    # Créer le simulateur
                    simulator = QuantumGravitySimulatorMobile(
                        grid_size=grid_size,
                        time_steps=time_steps,
                        precision=precision,
                        memory_limit_mb=memory_limit,
                        use_cache=True
                    )
                    
                    # Si la taille de grille a été ajustée, utiliser la nouvelle taille
                    actual_grid_size = simulator.grid_size
                    
                    # Mesurer les performances (moyenne sur num_runs)
                    durations = []
                    memory_usages = []
                    
                    for run in range(num_runs):
                        # Effacer le cache entre les runs
                        QuantumGravitySimulatorMobile.clear_cache()
                        
                        # Exécuter la simulation
                        start_time = time.time()
                        simulator.run_simulation_step()
                        duration = time.time() - start_time
                        
                        # Mesurer l'utilisation mémoire
                        memory_usage = simulator.get_memory_usage_mb()
                        
                        durations.append(duration)
                        memory_usages.append(memory_usage)
                    
                    # Calculer les moyennes
                    avg_duration = sum(durations) / len(durations)
                    avg_memory = sum(memory_usages) / len(memory_usages)
                    
                    # Enregistrer les résultats
                    config = {
                        "grid_size": grid_size,
                        "actual_grid_size": actual_grid_size,
                        "time_steps": time_steps,
                        "precision": precision,
                        "memory_limit": memory_limit
                    }
                    
                    performance = {
                        "avg_duration": avg_duration,
                        "min_duration": min(durations),
                        "max_duration": max(durations),
                        "durations": durations
                    }
                    
                    memory = {
                        "avg_usage": avg_memory,
                        "min_usage": min(memory_usages),
                        "max_usage": max(memory_usages),
                        "usages": memory_usages
                    }
                    
                    results["configurations"].append(config)
                    results["performance"].append(performance)
                    results["memory"].append(memory)
                    
                    logger.info(f"Résultats: Durée {avg_duration:.4f}s, Mémoire {avg_memory:.2f} MB")
                    
                except Exception as e:
                    logger.error(f"Erreur pour la configuration {grid_size}x{grid_size}, {precision}, {memory_limit} MB: {str(e)}")
                    # Ajouter un résultat d'erreur
                    results["configurations"].append({
                        "grid_size": grid_size,
                        "time_steps": time_steps,
                        "precision": precision,
                        "memory_limit": memory_limit,
                        "error": str(e)
                    })
                    results["performance"].append({})
                    results["memory"].append({})
    
    return results

def plot_benchmark_results(results, output_file="benchmark_mobile_results.png"):
    """
    Génère des graphiques pour visualiser les résultats du benchmark
    
    Args:
        results: Résultats du benchmark
        output_file: Fichier de sortie pour le graphique
    """
    grid_sizes = results["grid_sizes"]
    precisions = results["precisions"]
    
    # Préparer les données pour le tracé
    performance_data = {precision: [] for precision in precisions}
    memory_data = {precision: [] for precision in precisions}
    
    for i, config in enumerate(results["configurations"]):
        grid_size = config.get("grid_size")
        precision = config.get("precision")
        performance = results["performance"][i]
        memory = results["memory"][i]
        
        if not performance or not memory or "error" in config:
            # Ignorer les configurations avec erreur
            continue
        
        avg_duration = performance.get("avg_duration", 0)
        avg_memory = memory.get("avg_usage", 0)
        
        # On n'utilise que la première limite de mémoire pour simplifier le graphique
        if config.get("memory_limit") == results["memory_limits"][0]:
            performance_data[precision].append((grid_size, avg_duration))
            memory_data[precision].append((grid_size, avg_memory))
    
    # Créer le graphique
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Graphique des performances
    for precision, data in performance_data.items():
        if data:
            x, y = zip(*sorted(data))
            ax1.plot(x, y, marker='o', linestyle='-', label=precision)
    
    ax1.set_xlabel('Taille de grille')
    ax1.set_ylabel('Temps d\'exécution (s)')
    ax1.set_title('Performance par taille de grille et précision')
    ax1.grid(True)
    ax1.legend()
    
    # Graphique de l'utilisation mémoire
    for precision, data in memory_data.items():
        if data:
            x, y = zip(*sorted(data))
            ax2.plot(x, y, marker='o', linestyle='-', label=precision)
    
    ax2.set_xlabel('Taille de grille')
    ax2.set_ylabel('Utilisation mémoire (MB)')
    ax2.set_title('Empreinte mémoire par taille de grille et précision')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_file)
    logger.info(f"Graphique enregistré: {output_file}")

def main():
    """
    Fonction principale
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark du simulateur pour appareils mobiles")
    parser.add_argument("--output", type=str, default="benchmark_mobile_results.json", 
                      help="Fichier de sortie pour les résultats")
    parser.add_argument("--plot", type=str, default="benchmark_mobile_results.png",
                      help="Fichier de sortie pour le graphique")
    parser.add_argument("--runs", type=int, default=3,
                      help="Nombre d'exécutions par configuration")
    
    args = parser.parse_args()
    
    # Configurations de test optimisées pour les appareils mobiles
    grid_sizes = [8, 16, 32]  # Simplifié pour un test rapide
    time_steps = 4  # Valeur typique pour mobile
    precisions = ["float32", "float16", "int8"]
    memory_limits = [100]  # MB, simplifié pour un test rapide
    
    # Exécuter le benchmark
    logger.info("Démarrage du benchmark pour appareils mobiles")
    results = run_benchmark(
        grid_sizes=grid_sizes,
        time_steps=time_steps,
        precisions=precisions,
        memory_limits=memory_limits,
        num_runs=args.runs
    )
    
    # Sauvegarder les résultats
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Résultats enregistrés: {args.output}")
    
    # Tracer les graphiques
    plot_benchmark_results(results, output_file=args.plot)
    
    # Afficher un résumé
    logger.info("=== RÉSUMÉ DU BENCHMARK ===")
    for i, config in enumerate(results["configurations"]):
        performance = results["performance"][i]
        memory = results["memory"][i]
        
        if not performance or not memory or "error" in config:
            continue
            
        grid_size = config.get("grid_size")
        precision = config.get("precision")
        memory_limit = config.get("memory_limit")
        
        logger.info(f"Grille {grid_size}x{grid_size}, {precision}, {memory_limit} MB: "
                  f"{performance.get('avg_duration', 0):.4f}s, "
                  f"{memory.get('avg_usage', 0):.2f} MB")

if __name__ == "__main__":
    main()