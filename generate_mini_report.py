#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script pour générer un rapport visuel des résultats du mini-benchmark
"""

import os
import sys
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Constantes
OPTIMIZATION_DIRS = sorted(glob.glob("lr_optimization_*"), reverse=True)
LEARNING_DIRS = sorted(glob.glob("learning_results_*"), reverse=True)
OUTPUT_DIR = "mini_benchmark_report"

def ensure_output_dir():
    """Crée le répertoire de sortie s'il n'existe pas"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def load_optimization_results():
    """Charge les résultats d'optimisation des taux d'apprentissage"""
    results = []
    
    for opt_dir in OPTIMIZATION_DIRS[:3]:  # Considérer uniquement les 3 plus récents
        summary_file = os.path.join(opt_dir, "training_optimization_summary.json")
        if os.path.exists(summary_file):
            with open(summary_file, "r") as f:
                try:
                    data = json.load(f)
                    results.append(data)
                    print(f"Chargement des résultats d'optimisation depuis {summary_file}")
                except json.JSONDecodeError:
                    print(f"Erreur lors du décodage de {summary_file}")
    
    return results

def load_learning_results():
    """Charge les résultats d'analyse d'apprentissage"""
    results = []
    
    for learn_dir in LEARNING_DIRS[:3]:  # Considérer uniquement les 3 plus récents
        summary_file = os.path.join(learn_dir, "training_summary.json")
        if os.path.exists(summary_file):
            with open(summary_file, "r") as f:
                try:
                    data = json.load(f)
                    results.append(data)
                    print(f"Chargement des résultats d'apprentissage depuis {summary_file}")
                except json.JSONDecodeError:
                    print(f"Erreur lors du décodage de {summary_file}")
    
    return results

def plot_learning_rate_distribution(optimization_results):
    """Génère un graphique de la distribution des taux d'apprentissage"""
    plt.figure(figsize=(10, 6))
    
    if not optimization_results:
        plt.title("Aucun résultat d'optimisation trouvé")
        plt.savefig(os.path.join(OUTPUT_DIR, "learning_rate_distribution.png"))
        return
    
    # Fusionner les distributions de tous les résultats
    lr_distribution = {}
    for result in optimization_results:
        for lr, count in result.get("learning_rate_distribution", {}).items():
            lr_distribution[lr] = lr_distribution.get(lr, 0) + count
    
    # Créer le graphique
    if lr_distribution:
        labels = list(lr_distribution.keys())
        counts = list(lr_distribution.values())
        
        plt.bar(labels, counts, color='skyblue')
        plt.xlabel('Taux d\'apprentissage')
        plt.ylabel('Nombre de puzzles')
        plt.title('Distribution des taux d\'apprentissage optimaux')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Ajouter les valeurs sur les barres
        for i, v in enumerate(counts):
            plt.text(i, v + 0.1, str(v), ha='center')
        
        plt.savefig(os.path.join(OUTPUT_DIR, "learning_rate_distribution.png"))

def plot_loss_progression(learning_results):
    """Génère un graphique de la progression de la perte pendant l'apprentissage"""
    plt.figure(figsize=(10, 6))
    
    if not learning_results:
        plt.title("Aucun résultat d'apprentissage trouvé")
        plt.savefig(os.path.join(OUTPUT_DIR, "loss_progression.png"))
        return
    
    # Extraire les progressions de perte de tous les puzzles
    all_loss_progressions = []
    for result in learning_results:
        for puzzle_result in result.get("all_results", []):
            loss_progression = puzzle_result.get("loss_progression", [])
            if loss_progression:
                all_loss_progressions.append((puzzle_result.get("puzzle_id", "inconnu"), loss_progression))
    
    # Créer le graphique
    if all_loss_progressions:
        for puzzle_id, loss_progression in all_loss_progressions:
            epochs = list(range(1, len(loss_progression) + 1))
            plt.plot(epochs, loss_progression, label=f'Puzzle {puzzle_id}')
        
        plt.xlabel('Epochs')
        plt.ylabel('Perte (loss)')
        plt.title('Progression de la perte pendant l\'apprentissage')
        plt.grid(linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.savefig(os.path.join(OUTPUT_DIR, "loss_progression.png"))

def generate_metrics_summary(optimization_results, learning_results):
    """Génère une synthèse des métriques importantes"""
    with open(os.path.join(OUTPUT_DIR, "metrics_summary.md"), "w") as f:
        f.write("# Synthèse des Métriques du Mini-Benchmark Neurax2\n\n")
        f.write(f"*Généré le {datetime.now().strftime('%Y-%m-%d à %H:%M:%S')}*\n\n")
        
        f.write("## Optimisation des Taux d'Apprentissage\n\n")
        
        if optimization_results:
            # Calculer les moyennes sur tous les résultats
            total_puzzles = sum(r.get("total_puzzles", 0) for r in optimization_results)
            avg_lr = np.mean([r.get("average_best_learning_rate", 0) for r in optimization_results])
            avg_loss = np.mean([r.get("average_final_loss", 0) for r in optimization_results])
            success_count = sum(r.get("success_count", 0) for r in optimization_results)
            
            f.write(f"- **Puzzles analysés**: {total_puzzles}\n")
            f.write(f"- **Taux d'apprentissage moyen optimal**: {avg_lr:.6f}\n")
            f.write(f"- **Perte moyenne finale**: {avg_loss:.6f}\n")
            f.write(f"- **Puzzles réussis**: {success_count}/{total_puzzles} ({100 * success_count / total_puzzles if total_puzzles else 0:.1f}%)\n\n")
        else:
            f.write("*Aucun résultat d'optimisation trouvé*\n\n")
        
        f.write("## Analyse d'Apprentissage\n\n")
        
        if learning_results:
            # Calculer les moyennes sur tous les résultats
            total_puzzles = sum(r.get("total_puzzles", 0) for r in learning_results)
            converged_count = sum(r.get("converged_count", 0) for r in learning_results)
            avg_epochs = np.mean([r.get("average_epochs", 0) for r in learning_results if "average_epochs" in r])
            avg_loss = np.mean([r.get("average_final_loss", 0) for r in learning_results if "average_final_loss" in r])
            
            f.write(f"- **Puzzles analysés**: {total_puzzles}\n")
            f.write(f"- **Puzzles convergés**: {converged_count}/{total_puzzles} ({100 * converged_count / total_puzzles if total_puzzles else 0:.1f}%)\n")
            f.write(f"- **Nombre moyen d'epochs**: {avg_epochs:.1f}\n")
            f.write(f"- **Perte moyenne finale**: {avg_loss:.6f}\n\n")
        else:
            f.write("*Aucun résultat d'apprentissage trouvé*\n\n")
        
        f.write("## Conclusions\n\n")
        
        if optimization_results:
            f.write(f"- Le taux d'apprentissage optimal de {avg_lr:.6f} montre d'excellents résultats avec une perte finale moyenne de {avg_loss:.6f}.\n")
            f.write(f"- Le système atteint un taux de réussite de {100 * success_count / total_puzzles if total_puzzles else 0:.1f}% sur l'échantillon de puzzles testés.\n")
            
            # Recommandations basées sur les résultats
            if avg_lr > 0.15:
                f.write("- La valeur élevée du taux d'apprentissage optimal suggère que le système converge rapidement, ce qui est excellent pour les performances.\n")
            else:
                f.write("- La valeur modérée du taux d'apprentissage optimal suggère un bon équilibre entre vitesse et précision de convergence.\n")
        
        if learning_results and total_puzzles > 0:
            f.write(f"- L'analyse d'apprentissage montre que {100 * converged_count / total_puzzles:.1f}% des puzzles atteignent la convergence.\n")
            
            if avg_epochs < 50:
                f.write("- Le nombre moyen d'epochs relativement bas indique une bonne efficacité d'apprentissage.\n")
            else:
                f.write("- Le nombre moyen d'epochs élevé suggère des puzzles complexes nécessitant un apprentissage approfondi.\n")

def main():
    """Fonction principale"""
    print("Génération du rapport du mini-benchmark Neurax2...")
    
    # Créer le répertoire de sortie
    ensure_output_dir()
    
    # Charger les résultats
    optimization_results = load_optimization_results()
    learning_results = load_learning_results()
    
    # Générer les graphiques et résumés
    plot_learning_rate_distribution(optimization_results)
    plot_loss_progression(learning_results)
    generate_metrics_summary(optimization_results, learning_results)
    
    print(f"Rapport généré avec succès dans le répertoire {OUTPUT_DIR}")

if __name__ == "__main__":
    main()