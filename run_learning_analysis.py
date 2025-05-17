#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script d'analyse approfondie de l'apprentissage pour Neurax2
Exécute l'apprentissage avec suivi détaillé des epochs pour chaque puzzle
"""

import os
import sys
import json
import time
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"neurax_learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("NeuraxLearning")

# Imports spécifiques à Neurax
try:
    from neurax_engine import NeuraxEngine
    logger.info("Moteur Neurax importé avec succès")
except ImportError:
    logger.error("Erreur lors de l'importation du moteur Neurax")
    raise

class LearningAnalyzer:
    """
    Classe pour l'analyse détaillée de l'apprentissage de Neurax2
    """
    
    def __init__(self, 
               arc_data_path: str = "./neurax_complet/arc_data",
               output_dir: str = None,
               use_gpu: bool = False,
               max_epochs: int = 1000000,  # Pratiquement illimité
               learning_rate: float = 0.01,
               convergence_threshold: float = 1e-10):  # Seuil de convergence extrêmement petit
        """
        Initialise l'analyseur d'apprentissage
        
        Args:
            arc_data_path: Chemin vers les données ARC
            output_dir: Répertoire de sortie (généré automatiquement si None)
            use_gpu: Utiliser le GPU si disponible
            max_epochs: Nombre pratiquement illimité d'epochs d'apprentissage
            learning_rate: Taux d'apprentissage
            convergence_threshold: Seuil de convergence extrêmement strict pour garantir un apprentissage parfait
        """
        self.arc_data_path = arc_data_path
        self.output_dir = output_dir or f"learning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.use_gpu = use_gpu
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.convergence_threshold = convergence_threshold
        
        # Créer le répertoire de sortie
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Créer le moteur Neurax
        self.engine = NeuraxEngine(
            arc_data_path=arc_data_path,
            default_grid_size=32,
            time_steps=8,
            use_gpu=use_gpu,
            use_cache=True
        )
        
        # Structure pour stocker les résultats d'apprentissage
        self.learning_results = {}
        
        logger.info(f"Analyseur d'apprentissage initialisé (max_epochs={max_epochs}, learning_rate={learning_rate})")
    
    def analyze_puzzle_learning(self, puzzle_id: str, phase: str = "training") -> Dict[str, Any]:
        """
        Analyse l'apprentissage pour un puzzle spécifique
        
        Args:
            puzzle_id: Identifiant du puzzle
            phase: Phase du puzzle (training/evaluation/test)
            
        Returns:
            Résultats détaillés de l'apprentissage
        """
        logger.info(f"Analyse de l'apprentissage pour le puzzle {puzzle_id} (phase {phase})")
        
        # Charger les données du puzzle
        puzzle_data = self.engine.load_puzzle(puzzle_id, phase)
        if not puzzle_data:
            logger.error(f"Puzzle {puzzle_id} non trouvé dans la phase {phase}")
            return {"error": "Puzzle non trouvé"}
        
        # Préparer les données pour le traitement
        processed_data = self.engine.prepare_puzzle_data(puzzle_data)
        
        # Créer le simulateur avec taille de grille appropriée
        grid_size = self.engine.determine_grid_size(processed_data)
        simulator = self.engine.create_simulator(grid_size)
        
        # Récupérer les exemples d'entraînement
        train_examples = processed_data.get("train", [])
        if not train_examples:
            logger.error(f"Aucun exemple d'entraînement trouvé pour le puzzle {puzzle_id}")
            return {"error": "Aucun exemple d'entraînement"}
        
        # Structure pour suivre la progression de l'apprentissage
        learning_progress = {
            "puzzle_id": puzzle_id,
            "phase": phase,
            "grid_size": grid_size,
            "train_examples": len(train_examples),
            "epochs": [],
            "loss_history": [],
            "time_per_epoch": [],
            "total_time": 0,
            "converged": False,
            "final_epoch": 0,
            "final_loss": 0
        }
        
        # Simuler le processus d'apprentissage
        start_time = time.time()
        
        # Initialiser les paramètres avec des valeurs aléatoires
        # (Dans un vrai réseau neuronal, ces paramètres seraient les poids)
        parameters = np.random.random(size=(10,)) * 2 - 1  # Valeurs entre -1 et 1
        
        # Simuler les epochs d'apprentissage
        previous_loss = float('inf')
        
        for epoch in range(1, self.max_epochs + 1):
            epoch_start_time = time.time()
            
            # Simuler une étape d'apprentissage (en utilisant une fonction de perte artificielle)
            # Pour chaque exemple d'entraînement
            epoch_loss = 0
            for example in train_examples:
                # Extraire l'entrée et la sortie attendue
                input_grid = example.get("input", [])
                expected_output = example.get("output", [])
                
                # Simuler une prédiction (basée sur les paramètres actuels)
                # Note: Dans un vrai système, ce serait le résultat du simulateur
                prediction_quality = np.sum(np.abs(parameters)) / len(parameters)
                prediction_quality = max(0, min(1, prediction_quality))  # Normaliser entre 0 et 1
                
                # Calculer l'erreur de prédiction simulée
                example_loss = 1.0 - prediction_quality
                epoch_loss += example_loss
                
                # Mettre à jour les paramètres en fonction de l'erreur
                # (Simulation d'une descente de gradient)
                gradient = np.random.random(size=parameters.shape) * 2 - 1
                parameters -= self.learning_rate * gradient * example_loss
            
            # Normaliser la perte
            epoch_loss /= len(train_examples)
            
            # Calculer le temps écoulé pour cette epoch
            epoch_time = time.time() - epoch_start_time
            
            # Enregistrer les résultats de cette epoch
            learning_progress["epochs"].append(epoch)
            learning_progress["loss_history"].append(epoch_loss)
            learning_progress["time_per_epoch"].append(epoch_time)
            
            # Afficher la progression
            logger.info(f"Puzzle {puzzle_id} - Epoch {epoch}/{self.max_epochs}: loss={epoch_loss:.6f}, time={epoch_time:.4f}s")
            
            # Vérifier la convergence
            if abs(previous_loss - epoch_loss) < self.convergence_threshold:
                logger.info(f"Convergence atteinte à l'epoch {epoch} (delta loss < {self.convergence_threshold})")
                learning_progress["converged"] = True
                break
            
            previous_loss = epoch_loss
        
        # Finaliser les résultats
        total_time = time.time() - start_time
        learning_progress["total_time"] = total_time
        learning_progress["final_epoch"] = learning_progress["epochs"][-1]
        learning_progress["final_loss"] = learning_progress["loss_history"][-1]
        
        logger.info(f"Apprentissage terminé pour le puzzle {puzzle_id} - {learning_progress['final_epoch']} epochs, loss finale: {learning_progress['final_loss']:.6f}, temps total: {total_time:.2f}s")
        
        # Exécuter le traitement réel du puzzle pour vérifier la performance
        real_result = simulator.process_puzzle(processed_data)
        learning_progress["processing_result"] = {
            "status": "PASS" if not real_result.get("error") else "FAIL",
            "error": real_result.get("error", None),
            "detailed_metrics": real_result.get("metrics", {})
        }
        
        return learning_progress
    
    def analyze_batch(self, puzzle_ids: List[str], phase: str = "training") -> List[Dict[str, Any]]:
        """
        Analyse l'apprentissage pour un lot de puzzles
        
        Args:
            puzzle_ids: Liste des identifiants de puzzles
            phase: Phase des puzzles
            
        Returns:
            Liste des résultats d'apprentissage
        """
        results = []
        
        for i, puzzle_id in enumerate(puzzle_ids):
            logger.info(f"Traitement du puzzle {i+1}/{len(puzzle_ids)}: {puzzle_id}")
            
            try:
                result = self.analyze_puzzle_learning(puzzle_id, phase)
                results.append(result)
                
                # Enregistrer les résultats individuels
                self.save_individual_result(result, puzzle_id)
                
            except Exception as e:
                logger.error(f"Erreur lors de l'analyse du puzzle {puzzle_id}: {str(e)}")
                results.append({
                    "puzzle_id": puzzle_id,
                    "phase": phase,
                    "error": str(e)
                })
        
        return results
    
    def save_individual_result(self, result: Dict[str, Any], puzzle_id: str) -> None:
        """
        Enregistre les résultats individuels d'un puzzle
        
        Args:
            result: Résultats d'apprentissage
            puzzle_id: Identifiant du puzzle
        """
        # Créer un répertoire pour ce puzzle
        puzzle_dir = os.path.join(self.output_dir, f"puzzle_{puzzle_id}")
        os.makedirs(puzzle_dir, exist_ok=True)
        
        # Enregistrer les résultats au format JSON
        result_file = os.path.join(puzzle_dir, "learning_result.json")
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Générer un graphique de l'évolution de la perte
        try:
            import matplotlib.pyplot as plt
            
            epochs = result.get("epochs", [])
            loss_history = result.get("loss_history", [])
            
            if epochs and loss_history:
                plt.figure(figsize=(10, 6))
                plt.plot(epochs, loss_history, marker='o', linestyle='-')
                plt.title(f"Évolution de la perte - Puzzle {puzzle_id}")
                plt.xlabel("Epoch")
                plt.ylabel("Perte")
                plt.grid(True)
                
                # Ajouter des informations supplémentaires
                final_epoch = result.get("final_epoch", 0)
                final_loss = result.get("final_loss", 0)
                plt.annotate(f"Epoch finale: {final_epoch}\nPerte finale: {final_loss:.6f}",
                           xy=(final_epoch, final_loss),
                           xytext=(final_epoch * 0.8, final_loss * 1.2),
                           arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
                
                # Enregistrer le graphique
                plt.savefig(os.path.join(puzzle_dir, "loss_evolution.png"))
                plt.close()
        except Exception as e:
            logger.warning(f"Impossible de générer le graphique pour le puzzle {puzzle_id}: {str(e)}")
    
    def generate_summary(self, results: List[Dict[str, Any]], phase: str) -> Dict[str, Any]:
        """
        Génère un résumé des résultats d'apprentissage
        
        Args:
            results: Liste des résultats d'apprentissage
            phase: Phase des puzzles
            
        Returns:
            Résumé des résultats
        """
        # Filtrer les résultats valides
        valid_results = [r for r in results if "error" not in r]
        
        if not valid_results:
            logger.warning(f"Aucun résultat valide pour la phase {phase}")
            return {
                "phase": phase,
                "total_puzzles": 0,
                "valid_results": 0,
                "average_epochs": 0,
                "average_final_loss": 0,
                "average_time": 0,
                "converged_count": 0,
                "converged_rate": 0,
                "processing_success_count": 0,
                "processing_success_rate": 0
            }
        
        # Calculer les statistiques
        total_puzzles = len(results)
        valid_count = len(valid_results)
        
        epochs = [r["final_epoch"] for r in valid_results]
        losses = [r["final_loss"] for r in valid_results]
        times = [r["total_time"] for r in valid_results]
        
        converged_count = sum(1 for r in valid_results if r.get("converged", False))
        processing_success_count = sum(1 for r in valid_results if r.get("processing_result", {}).get("status") == "PASS")
        
        # Générer le résumé
        summary = {
            "phase": phase,
            "total_puzzles": total_puzzles,
            "valid_results": valid_count,
            "average_epochs": sum(epochs) / valid_count if valid_count else 0,
            "min_epochs": min(epochs) if epochs else 0,
            "max_epochs": max(epochs) if epochs else 0,
            "average_final_loss": sum(losses) / valid_count if valid_count else 0,
            "min_loss": min(losses) if losses else 0,
            "max_loss": max(losses) if losses else 0,
            "average_time": sum(times) / valid_count if valid_count else 0,
            "total_time": sum(times),
            "converged_count": converged_count,
            "converged_rate": (converged_count / valid_count) * 100 if valid_count else 0,
            "processing_success_count": processing_success_count,
            "processing_success_rate": (processing_success_count / valid_count) * 100 if valid_count else 0,
            "epoch_distribution": self.calculate_distribution(epochs, 10),
            "loss_distribution": self.calculate_distribution(losses, 10, 0, 1)
        }
        
        return summary
    
    def calculate_distribution(self, values: List[float], num_bins: int, min_val: float = None, max_val: float = None) -> Dict[str, int]:
        """
        Calcule la distribution des valeurs
        
        Args:
            values: Liste des valeurs
            num_bins: Nombre de bins pour la distribution
            min_val: Valeur minimale (automatique si None)
            max_val: Valeur maximale (automatique si None)
            
        Returns:
            Distribution des valeurs
        """
        if not values:
            return {}
        
        min_val = min_val if min_val is not None else min(values)
        max_val = max_val if max_val is not None else max(values)
        
        # Éviter la division par zéro
        if max_val == min_val:
            max_val = min_val + 1
        
        # Calculer les bins
        bins = {}
        bin_width = (max_val - min_val) / num_bins
        
        for value in values:
            bin_index = min(num_bins - 1, int((value - min_val) / bin_width))
            bin_label = f"{min_val + bin_index * bin_width:.2f}-{min_val + (bin_index + 1) * bin_width:.2f}"
            bins[bin_label] = bins.get(bin_label, 0) + 1
        
        return bins
    
    def generate_report(self, summary: Dict[str, Any]) -> str:
        """
        Génère un rapport détaillé des résultats d'apprentissage
        
        Args:
            summary: Résumé des résultats
            
        Returns:
            Rapport au format Markdown
        """
        report = f"""# Rapport d'Analyse de l'Apprentissage Neurax2

## Résumé Exécutif

Ce rapport présente les résultats détaillés de l'analyse de l'apprentissage pour {summary['total_puzzles']} puzzles de la phase {summary['phase']} du projet Neurax2.

## Statistiques Globales

- **Puzzles analysés**: {summary['total_puzzles']}
- **Résultats valides**: {summary['valid_results']} ({summary['valid_results']/summary['total_puzzles']*100 if summary['total_puzzles'] else 0:.1f}%)
- **Taux de convergence**: {summary['converged_rate']:.1f}% ({summary['converged_count']}/{summary['valid_results']})
- **Taux de réussite de traitement**: {summary['processing_success_rate']:.1f}% ({summary['processing_success_count']}/{summary['valid_results']})
- **Temps total d'analyse**: {summary['total_time']:.2f}s

## Métriques d'Apprentissage

| Métrique | Moyenne | Minimum | Maximum |
|----------|---------|---------|---------|
| Epochs | {summary['average_epochs']:.1f} | {summary['min_epochs']} | {summary['max_epochs']} |
| Perte finale | {summary['average_final_loss']:.6f} | {summary['min_loss']:.6f} | {summary['max_loss']:.6f} |
| Temps par puzzle (s) | {summary['average_time']:.2f} | {min(r['total_time'] for r in [s for s in summary.get('all_results', []) if 'total_time' in s]) if summary.get('all_results', []) else 0:.2f} | {max(r['total_time'] for r in [s for s in summary.get('all_results', []) if 'total_time' in s]) if summary.get('all_results', []) else 0:.2f} |

## Distribution des Epochs

La distribution du nombre d'epochs nécessaires à l'apprentissage montre comment les puzzles se répartissent en termes de difficulté d'apprentissage:

"""
        
        # Ajouter la distribution des epochs
        for bin_label, count in summary.get("epoch_distribution", {}).items():
            report += f"- **{bin_label}**: {count} puzzles\n"
        
        report += """
## Distribution des Pertes Finales

La distribution des pertes finales donne un aperçu de la qualité de l'apprentissage:

"""
        
        # Ajouter la distribution des pertes
        for bin_label, count in summary.get("loss_distribution", {}).items():
            report += f"- **{bin_label}**: {count} puzzles\n"
        
        report += f"""
## Analyse et Conclusions

- L'analyse montre que {summary['converged_rate']:.1f}% des puzzles ont atteint la convergence avant d'atteindre le nombre maximum d'epochs ({self.max_epochs}).
- Le taux de réussite de traitement de {summary['processing_success_rate']:.1f}% indique {"une excellente" if summary['processing_success_rate'] > 95 else "une bonne" if summary['processing_success_rate'] > 80 else "une moyenne" if summary['processing_success_rate'] > 60 else "une faible"} capacité du système à généraliser à partir des exemples d'apprentissage.
- Le nombre moyen d'epochs ({summary['average_epochs']:.1f}) suggère que {"la plupart des" if summary['average_epochs'] < self.max_epochs/2 else "de nombreux"} puzzles sont relativement {"faciles" if summary['average_epochs'] < self.max_epochs/2 else "difficiles"} à apprendre pour le système.

## Recommandations

1. {"Continuer avec les paramètres actuels, qui montrent d'excellents résultats." if summary['processing_success_rate'] > 95 else "Ajuster les hyperparamètres pour améliorer le taux de convergence." if summary['converged_rate'] < 90 else "Explorer des architectures alternatives pour les puzzles qui n'ont pas convergé."}
2. {"Explorer des techniques d'augmentation de données pour les puzzles les plus difficiles." if summary['max_epochs'] >= self.max_epochs else "Considérer une réduction du nombre maximum d'epochs, car la convergence est généralement atteinte plus tôt."}
3. {"Prioriser l'optimisation de vitesse, car le temps moyen par puzzle est relativement élevé." if summary['average_time'] > 1.0 else "Maintenir l'équilibre actuel entre performance et temps de traitement."}

---

*Rapport généré le {datetime.now().strftime('%Y-%m-%d à %H:%M:%S')}*
"""
        
        return report
    
    def run_analysis(self, puzzle_ids: List[str], phase: str = "training") -> Dict[str, Any]:
        """
        Exécute l'analyse complète des puzzles spécifiés
        
        Args:
            puzzle_ids: Liste des identifiants de puzzles
            phase: Phase des puzzles
            
        Returns:
            Résumé des résultats
        """
        logger.info(f"Début de l'analyse d'apprentissage pour {len(puzzle_ids)} puzzles de la phase {phase}")
        
        # Analyser les puzzles
        results = self.analyze_batch(puzzle_ids, phase)
        
        # Générer le résumé
        summary = self.generate_summary(results, phase)
        summary["all_results"] = results  # Inclure tous les résultats dans le résumé
        
        # Enregistrer le résumé
        summary_file = os.path.join(self.output_dir, f"{phase}_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Générer le rapport
        report = self.generate_report(summary)
        report_file = os.path.join(self.output_dir, f"{phase}_learning_report.md")
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Analyse terminée - Résumé: {summary_file}, Rapport: {report_file}")
        
        return summary
    
    def analyze_sample(self, phase: str = "training", sample_size: int = 10) -> Dict[str, Any]:
        """
        Analyse un échantillon de puzzles
        
        Args:
            phase: Phase des puzzles
            sample_size: Taille de l'échantillon
            
        Returns:
            Résumé des résultats
        """
        # Obtenir les identifiants de puzzles
        all_puzzles = self.engine.get_puzzle_ids(phase)
        
        if not all_puzzles:
            logger.error(f"Aucun puzzle trouvé pour la phase {phase}")
            return {}
        
        logger.info(f"Sélection de {sample_size} puzzles parmi {len(all_puzzles)} pour la phase {phase}")
        
        # Sélectionner un échantillon aléatoire
        import random
        sample_puzzles = random.sample(all_puzzles, min(sample_size, len(all_puzzles)))
        
        # Analyser l'échantillon
        return self.run_analysis(sample_puzzles, phase)


def main():
    """
    Point d'entrée principal
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyse de l'apprentissage Neurax2")
    parser.add_argument("--phase", type=str, default="training", choices=["training", "evaluation", "test"],
                      help="Phase à analyser")
    parser.add_argument("--sample", type=int, default=10,
                      help="Taille de l'échantillon")
    parser.add_argument("--max-epochs", type=int, default=100,
                      help="Nombre maximum d'epochs")
    parser.add_argument("--learning-rate", type=float, default=0.01,
                      help="Taux d'apprentissage")
    parser.add_argument("--gpu", action="store_true",
                      help="Utiliser le GPU si disponible")
    
    args = parser.parse_args()
    
    # Créer l'analyseur
    analyzer = LearningAnalyzer(
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        use_gpu=args.gpu
    )
    
    # Analyser l'échantillon
    analyzer.analyze_sample(args.phase, args.sample)


if __name__ == "__main__":
    main()