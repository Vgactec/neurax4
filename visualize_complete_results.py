#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de visualisation des résultats complets des tests sur les 1360 puzzles ARC
Génère des graphiques et visualisations détaillées
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Configuration des graphiques
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12

def load_results(results_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Charge les résultats des tests ARC
    
    Args:
        results_dir: Répertoire contenant les résultats (ou None pour détecter automatiquement)
        
    Returns:
        Dictionnaire avec les résultats
    """
    # Trouver le répertoire de résultats le plus récent
    if results_dir is None:
        dirs = sorted(glob.glob('arc_results_*'), key=os.path.getmtime, reverse=True)
        if not dirs:
            print("Aucun répertoire de résultats trouvé.")
            return {}
        results_dir = dirs[0]
    
    print(f"Chargement des résultats depuis {results_dir}")
    
    # Structure pour stocker les résultats
    results = {
        "global": {},
        "phases": {},
        "puzzles": {}
    }
    
    # Charger le résumé global
    global_summary_files = glob.glob(f"{results_dir}/*global*summary.json")
    if global_summary_files:
        with open(global_summary_files[0], 'r') as f:
            results["global"] = json.load(f)
    
    # Charger les résumés par phase
    for phase in ["training", "evaluation", "test"]:
        phase_summary_files = glob.glob(f"{results_dir}/*{phase}*summary.json")
        if phase_summary_files:
            with open(phase_summary_files[0], 'r') as f:
                results["phases"][phase] = json.load(f)
        
        # Charger les résultats individuels des puzzles
        phase_dir = os.path.join(results_dir, phase)
        if os.path.exists(phase_dir):
            for puzzle_dir in os.listdir(phase_dir):
                puzzle_result_file = os.path.join(phase_dir, puzzle_dir, "results.json")
                if os.path.exists(puzzle_result_file):
                    with open(puzzle_result_file, 'r') as f:
                        puzzle_result = json.load(f)
                        results["puzzles"][puzzle_dir] = puzzle_result
    
    # Charger les résultats CSV
    csv_files = glob.glob(f"{results_dir}/*.csv")
    for csv_file in csv_files:
        phase = None
        for p in ["training", "evaluation", "test"]:
            if p in os.path.basename(csv_file):
                phase = p
                break
        
        if phase:
            try:
                df = pd.read_csv(csv_file)
                results[f"{phase}_csv"] = df
            except Exception as e:
                print(f"Erreur lors du chargement de {csv_file}: {str(e)}")
    
    return results

def create_puzzle_success_plot(results: Dict[str, Any], output_file: str = "puzzles_success_rate.png") -> None:
    """
    Crée un graphique du taux de réussite des puzzles par phase
    
    Args:
        results: Résultats des tests
        output_file: Fichier de sortie pour le graphique
    """
    global_summary = results.get("global", {})
    phases_data = global_summary.get("phases", {})
    
    if not phases_data:
        print("Aucune donnée de phase trouvée.")
        return
    
    # Préparer les données
    phases = []
    totals = []
    successes = []
    rates = []
    
    for phase, data in phases_data.items():
        phases.append(phase.capitalize())
        totals.append(data.get("total", 0))
        successes.append(data.get("success", 0))
        rates.append(data.get("success_rate", 0))
    
    # Ajouter le total global
    phases.append("Total")
    totals.append(global_summary.get("total_puzzles", 0))
    successes.append(global_summary.get("processing_success_count", 0))
    rates.append(global_summary.get("processing_success_rate", 0))
    
    # Créer le graphique
    fig, ax1 = plt.subplots()
    
    x = np.arange(len(phases))
    width = 0.35
    
    # Barres pour le nombre de puzzles
    rects1 = ax1.bar(x - width/2, totals, width, label='Total Puzzles', color='skyblue')
    rects2 = ax1.bar(x + width/2, successes, width, label='Puzzles Réussis', color='lightgreen')
    
    # Axe pour le taux de réussite
    ax2 = ax1.twinx()
    line = ax2.plot(x, rates, 'ro-', label='Taux de Réussite (%)', linewidth=2)
    
    # Configurer les axes
    ax1.set_xlabel('Phase')
    ax1.set_ylabel('Nombre de Puzzles')
    ax2.set_ylabel('Taux de Réussite (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(phases)
    ax2.set_ylim(0, 105)
    
    # Ajouter les valeurs sur les barres
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax1.annotate(f'{height}',
                       xy=(rect.get_x() + rect.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    add_labels(rects1)
    add_labels(rects2)
    
    # Ajouter les taux de réussite
    for i, rate in enumerate(rates):
        ax2.annotate(f'{rate:.1f}%',
                   xy=(x[i], rate),
                   xytext=(0, -15),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   color='darkred')
    
    # Légende
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left')
    
    # Titre
    plt.title('Performances de Neurax2 sur les 1360 Puzzles ARC')
    
    # Enregistrer le graphique
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"Graphique des taux de réussite enregistré: {output_file}")

def create_learning_rate_plot(results: Dict[str, Any], output_file: str = "learning_rates_distribution.png") -> None:
    """
    Crée un graphique de la distribution des taux d'apprentissage optimaux
    
    Args:
        results: Résultats des tests
        output_file: Fichier de sortie pour le graphique
    """
    # Collecter les taux d'apprentissage optimaux
    learning_rates = {}
    
    # Vérifier dans les résumés de phase
    for phase, phase_data in results.get("phases", {}).items():
        lr_distribution = phase_data.get("learning_rate_distribution", {})
        if lr_distribution:
            for lr, count in lr_distribution.items():
                learning_rates[float(lr)] = learning_rates.get(float(lr), 0) + count
    
    # Vérifier dans les résultats individuels des puzzles
    for puzzle_id, puzzle_data in results.get("puzzles", {}).items():
        best_lr = puzzle_data.get("best_learning_rate")
        if best_lr is not None:
            learning_rates[best_lr] = learning_rates.get(best_lr, 0) + 1
    
    if not learning_rates:
        print("Aucune donnée sur les taux d'apprentissage trouvée.")
        return
    
    # Trier les taux
    sorted_lrs = sorted(learning_rates.items())
    lr_values = [lr for lr, _ in sorted_lrs]
    lr_counts = [count for _, count in sorted_lrs]
    
    # Créer le graphique
    fig, ax = plt.subplots()
    
    bars = ax.bar(lr_values, lr_counts, color='skyblue')
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom')
    
    # Calculer et afficher la moyenne
    average_lr = sum(lr * count for lr, count in sorted_lrs) / sum(lr_counts)
    
    ax.axvline(x=average_lr, color='red', linestyle='--')
    ax.annotate(f'Moyenne: {average_lr:.4f}',
               xy=(average_lr, max(lr_counts) * 0.8),
               xytext=(5, 0),
               textcoords="offset points",
               ha='left', va='center',
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
    
    # Configurer les axes
    ax.set_xlabel('Taux d\'Apprentissage')
    ax.set_ylabel('Nombre de Puzzles')
    ax.set_title('Distribution des Taux d\'Apprentissage Optimaux')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"Graphique des taux d'apprentissage enregistré: {output_file}")

def create_epochs_distribution_plot(results: Dict[str, Any], output_file: str = "epochs_distribution.png") -> None:
    """
    Crée un graphique de la distribution du nombre d'epochs nécessaires
    
    Args:
        results: Résultats des tests
        output_file: Fichier de sortie pour le graphique
    """
    # Collecter le nombre d'epochs
    epochs_data = []
    
    # Extraire depuis les résultats individuels des puzzles
    for puzzle_id, puzzle_data in results.get("puzzles", {}).items():
        epochs = puzzle_data.get("best_epochs")
        if epochs is not None:
            epochs_data.append(epochs)
    
    if not epochs_data:
        print("Aucune donnée sur le nombre d'epochs trouvée.")
        return
    
    # Créer le graphique
    fig, ax = plt.subplots()
    
    # Histogramme
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500, 1000, max(epochs_data) + 1]
    n, bins, patches = ax.hist(epochs_data, bins=bins, color='skyblue', edgecolor='black')
    
    # Ajouter les valeurs sur les barres
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    for i, count in enumerate(n):
        if count > 0:
            ax.annotate(f'{int(count)}',
                       xy=(bin_centers[i], count),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    # Configurer les axes
    ax.set_xlabel('Nombre d\'Epochs')
    ax.set_ylabel('Nombre de Puzzles')
    ax.set_title('Distribution du Nombre d\'Epochs Nécessaires à la Convergence')
    
    # Échelle logarithmique pour mieux voir les valeurs extrêmes
    ax.set_xscale('log')
    
    # Ligne verticale pour la moyenne
    mean_epochs = np.mean(epochs_data)
    ax.axvline(x=mean_epochs, color='red', linestyle='--')
    ax.annotate(f'Moyenne: {mean_epochs:.1f}',
               xy=(mean_epochs, max(n) * 0.8),
               xytext=(5, 0),
               textcoords="offset points",
               ha='left' if mean_epochs < max(epochs_data)/2 else 'right',
               va='center',
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"Graphique de la distribution des epochs enregistré: {output_file}")

def create_loss_vs_epochs_plot(results: Dict[str, Any], output_file: str = "loss_vs_epochs.png") -> None:
    """
    Crée un graphique de la perte finale en fonction du nombre d'epochs
    
    Args:
        results: Résultats des tests
        output_file: Fichier de sortie pour le graphique
    """
    # Collecter les données
    epochs_data = []
    loss_data = []
    learning_rates = []
    
    # Extraire depuis les résultats individuels des puzzles
    for puzzle_id, puzzle_data in results.get("puzzles", {}).items():
        epochs = puzzle_data.get("best_epochs")
        loss = puzzle_data.get("best_loss")
        lr = puzzle_data.get("best_learning_rate")
        
        if epochs is not None and loss is not None and lr is not None:
            epochs_data.append(epochs)
            loss_data.append(loss)
            learning_rates.append(lr)
    
    if not epochs_data:
        print("Aucune donnée sur les epochs et pertes trouvée.")
        return
    
    # Créer le graphique
    fig, ax = plt.subplots()
    
    # Normaliser les taux d'apprentissage pour la couleur
    norm = Normalize(vmin=min(learning_rates), vmax=max(learning_rates))
    cmap = cm.viridis
    
    # Scatter plot avec couleur selon le taux d'apprentissage
    scatter = ax.scatter(epochs_data, loss_data, c=learning_rates, cmap=cmap, alpha=0.7, s=50)
    
    # Ajouter une barre de couleur
    cbar = plt.colorbar(scatter)
    cbar.set_label('Taux d\'Apprentissage')
    
    # Configurer les axes
    ax.set_xlabel('Nombre d\'Epochs')
    ax.set_ylabel('Perte Finale')
    ax.set_title('Perte Finale en Fonction du Nombre d\'Epochs et du Taux d\'Apprentissage')
    
    # Échelle logarithmique pour mieux voir les valeurs
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"Graphique de la perte vs epochs enregistré: {output_file}")

def create_grid_size_vs_time_plot(results: Dict[str, Any], output_file: str = "grid_size_vs_time.png") -> None:
    """
    Crée un graphique du temps de traitement en fonction de la taille de grille
    
    Args:
        results: Résultats des tests
        output_file: Fichier de sortie pour le graphique
    """
    # Collecter les données
    grid_sizes = []
    proc_times = []
    
    # Extraire depuis les résultats individuels des puzzles
    for puzzle_id, puzzle_data in results.get("puzzles", {}).items():
        grid_size = puzzle_data.get("grid_size")
        proc_time = puzzle_data.get("processing_result", {}).get("processing_time")
        
        if grid_size is not None and proc_time is not None:
            grid_sizes.append(grid_size)
            proc_times.append(proc_time * 1000)  # Convertir en ms
    
    if not grid_sizes:
        print("Aucune donnée sur les tailles de grille et temps de traitement trouvée.")
        return
    
    # Créer le graphique
    fig, ax = plt.subplots()
    
    # Scatter plot
    ax.scatter(grid_sizes, proc_times, alpha=0.5, s=30)
    
    # Ajouter une ligne de tendance
    z = np.polyfit(grid_sizes, proc_times, 2)
    p = np.poly1d(z)
    x_range = np.linspace(min(grid_sizes), max(grid_sizes), 100)
    ax.plot(x_range, p(x_range), "r--", label=f"Tendance: {z[0]:.4f}x² + {z[1]:.4f}x + {z[2]:.4f}")
    
    # Configurer les axes
    ax.set_xlabel('Taille de Grille')
    ax.set_ylabel('Temps de Traitement (ms)')
    ax.set_title('Temps de Traitement en Fonction de la Taille de Grille')
    
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"Graphique de la taille de grille vs temps enregistré: {output_file}")

def generate_summary_report(results: Dict[str, Any], output_file: str = "visualisation_resultats.md") -> None:
    """
    Génère un rapport de synthèse des résultats
    
    Args:
        results: Résultats des tests
        output_file: Fichier de sortie pour le rapport
    """
    global_summary = results.get("global", {})
    phases_data = global_summary.get("phases", {})
    
    with open(output_file, 'w') as f:
        f.write("# Visualisation des Résultats - Neurax2 sur 1360 Puzzles ARC\n\n")
        
        f.write("## Résumé des Performances\n\n")
        
        if global_summary:
            f.write(f"- **Puzzles testés**: {global_summary.get('total_puzzles', 0)}\n")
            f.write(f"- **Puzzles valides**: {global_summary.get('valid_puzzles', 0)}\n")
            f.write(f"- **Puzzles réussis**: {global_summary.get('processing_success_count', 0)}\n")
            f.write(f"- **Taux de réussite global**: {global_summary.get('processing_success_rate', 0):.1f}%\n\n")
        
        if phases_data:
            f.write("### Performances par Phase\n\n")
            
            f.write("| Phase | Puzzles | Réussis | Taux de Réussite |\n")
            f.write("|-------|---------|---------|------------------|\n")
            
            for phase, data in phases_data.items():
                f.write(f"| {phase.capitalize()} | {data.get('total', 0)} | {data.get('success', 0)} | {data.get('success_rate', 0):.1f}% |\n")
            
            f.write("\n")
        
        # Ajouter les liens vers les graphiques
        f.write("## Visualisations\n\n")
        
        graphs = [
            ("puzzles_success_rate.png", "Taux de Réussite par Phase"),
            ("learning_rates_distribution.png", "Distribution des Taux d'Apprentissage Optimaux"),
            ("epochs_distribution.png", "Distribution du Nombre d'Epochs"),
            ("loss_vs_epochs.png", "Perte Finale vs Nombre d'Epochs"),
            ("grid_size_vs_time.png", "Temps de Traitement vs Taille de Grille")
        ]
        
        for graph_file, graph_title in graphs:
            f.write(f"### {graph_title}\n\n")
            f.write(f"![{graph_title}]({graph_file})\n\n")
        
        # Analyse
        f.write("## Analyse des Résultats\n\n")
        
        # Collecter les métriques pour l'analyse
        puzzles_data = results.get("puzzles", {})
        
        epochs_data = [p.get("best_epochs") for p in puzzles_data.values() if p.get("best_epochs") is not None]
        loss_data = [p.get("best_loss") for p in puzzles_data.values() if p.get("best_loss") is not None]
        lr_data = [p.get("best_learning_rate") for p in puzzles_data.values() if p.get("best_learning_rate") is not None]
        grid_sizes = [p.get("grid_size") for p in puzzles_data.values() if p.get("grid_size") is not None]
        
        if epochs_data:
            f.write("### Convergence\n\n")
            f.write(f"- **Nombre moyen d'epochs**: {np.mean(epochs_data):.1f}\n")
            f.write(f"- **Médiane d'epochs**: {np.median(epochs_data):.1f}\n")
            f.write(f"- **Minimum d'epochs**: {min(epochs_data)}\n")
            f.write(f"- **Maximum d'epochs**: {max(epochs_data)}\n\n")
        
        if loss_data:
            f.write("### Perte Finale\n\n")
            f.write(f"- **Perte moyenne**: {np.mean(loss_data):.6f}\n")
            f.write(f"- **Perte médiane**: {np.median(loss_data):.6f}\n")
            f.write(f"- **Perte minimale**: {min(loss_data):.6f}\n")
            f.write(f"- **Perte maximale**: {max(loss_data):.6f}\n\n")
        
        if lr_data:
            f.write("### Taux d'Apprentissage\n\n")
            f.write(f"- **Taux moyen**: {np.mean(lr_data):.6f}\n")
            f.write(f"- **Taux médian**: {np.median(lr_data):.6f}\n")
            
            # Calculer la distribution des taux
            lr_counts = {}
            for lr in lr_data:
                lr_counts[lr] = lr_counts.get(lr, 0) + 1
            
            f.write("\n**Distribution des taux**:\n\n")
            for lr, count in sorted(lr_counts.items()):
                f.write(f"- **{lr}**: {count} puzzles ({count/len(lr_data)*100:.1f}%)\n")
            
            f.write("\n")
        
        if grid_sizes:
            f.write("### Tailles de Grille\n\n")
            f.write(f"- **Taille moyenne**: {np.mean(grid_sizes):.1f}\n")
            f.write(f"- **Taille médiane**: {np.median(grid_sizes):.1f}\n")
            f.write(f"- **Taille minimale**: {min(grid_sizes)}\n")
            f.write(f"- **Taille maximale**: {max(grid_sizes)}\n\n")
            
            # Calculer la distribution des tailles
            grid_counts = {}
            for size in grid_sizes:
                grid_counts[size] = grid_counts.get(size, 0) + 1
            
            f.write("\n**Distribution des tailles**:\n\n")
            for size, count in sorted(grid_counts.items()):
                f.write(f"- **{size}x{size}**: {count} puzzles ({count/len(grid_sizes)*100:.1f}%)\n")
            
            f.write("\n")
        
        # Conclusion
        f.write("## Conclusion\n\n")
        
        success_rate = global_summary.get("processing_success_rate", 0)
        
        if success_rate == 100:
            f.write("L'analyse montre que Neurax2 a atteint son objectif de 100% de réussite sur l'ensemble des 1360 puzzles ARC. La suppression des limites d'epochs et l'optimisation individuelle des taux d'apprentissage ont permis une convergence parfaite pour tous les puzzles, garantissant ainsi une capacité de généralisation optimale du système.")
        elif success_rate > 95:
            f.write(f"L'analyse montre que Neurax2 a atteint un excellent taux de réussite de {success_rate:.1f}% sur l'ensemble des 1360 puzzles ARC. La suppression des limites d'epochs et l'optimisation individuelle des taux d'apprentissage ont contribué significativement à ces résultats. Pour atteindre 100% de réussite, des ajustements spécifiques pourraient être nécessaires pour les puzzles restants.")
        elif success_rate > 80:
            f.write(f"L'analyse montre que Neurax2 a atteint un bon taux de réussite de {success_rate:.1f}% sur l'ensemble des 1360 puzzles ARC. Bien que les limites d'epochs aient été supprimées, certains puzzles nécessitent encore des optimisations supplémentaires. Une analyse détaillée des échecs permettrait d'identifier les ajustements nécessaires pour atteindre l'objectif de 100% de réussite.")
        else:
            f.write(f"L'analyse montre que Neurax2 a atteint un taux de réussite de {success_rate:.1f}% sur l'ensemble des 1360 puzzles ARC. Des améliorations significatives sont nécessaires pour atteindre l'objectif de 100% de réussite. Une analyse approfondie des échecs et une révision de l'approche d'apprentissage seraient bénéfiques.")
        
        f.write("\n\n---\n\n")
        f.write(f"*Rapport généré le {datetime.now().strftime('%Y-%m-%d à %H:%M:%S')}*\n")
    
    print(f"Rapport de synthèse enregistré: {output_file}")

def main():
    """
    Fonction principale
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualisation des résultats des tests ARC complets")
    parser.add_argument("--dir", type=str, default=None,
                      help="Répertoire contenant les résultats (par défaut: détection automatique)")
    
    args = parser.parse_args()
    
    # Charger les résultats
    results = load_results(args.dir)
    
    if not results:
        print("Aucun résultat trouvé.")
        return
    
    # Créer les visualisations
    create_puzzle_success_plot(results)
    create_learning_rate_plot(results)
    create_epochs_distribution_plot(results)
    create_loss_vs_epochs_plot(results)
    create_grid_size_vs_time_plot(results)
    
    # Générer le rapport de synthèse
    generate_summary_report(results)
    
    print("Visualisation des résultats terminée.")

if __name__ == "__main__":
    main()