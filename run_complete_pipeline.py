#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script d'exécution de la chaîne complète de traitement Neurax2
Exécute l'apprentissage, la validation et la préparation pour Kaggle
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"neurax_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("NeuraxPipeline")

class NeuraxPipeline:
    """
    Classe pour l'exécution de la chaîne complète de traitement Neurax2
    """
    
    def __init__(self, 
               arc_data_path: str = "./neurax_complet/arc_data",
               output_dir: str = None,
               use_gpu: bool = False,
               optimize_learning: bool = True,
               learning_rate: float = 0.1,
               max_epochs: int = 100,
               kaggle_user: str = "ndarray2000",
               kaggle_key: str = "5354ea3f21950428c738b880332b0a5e"):
        """
        Initialise la chaîne de traitement
        
        Args:
            arc_data_path: Chemin vers les données ARC
            output_dir: Répertoire de sortie (généré automatiquement si None)
            use_gpu: Utiliser le GPU si disponible
            optimize_learning: Optimiser le taux d'apprentissage
            learning_rate: Taux d'apprentissage par défaut
            max_epochs: Nombre maximum d'epochs
            kaggle_user: Nom d'utilisateur Kaggle
            kaggle_key: Clé API Kaggle
        """
        self.arc_data_path = arc_data_path
        self.output_dir = output_dir or f"neurax_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.use_gpu = use_gpu
        self.optimize_learning = optimize_learning
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.kaggle_user = kaggle_user
        self.kaggle_key = kaggle_key
        
        # Créer le répertoire de sortie
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Pipeline Neurax initialisée (GPU: {use_gpu}, Optimisation: {optimize_learning}, LR: {learning_rate})")
    
    def ensure_script_executable(self, script_path: str) -> bool:
        """
        S'assure qu'un script est exécutable
        
        Args:
            script_path: Chemin vers le script
            
        Returns:
            True si le script est exécutable, False sinon
        """
        if not os.path.exists(script_path):
            logger.error(f"Le script {script_path} n'existe pas")
            return False
        
        # Rendre le script exécutable si nécessaire
        if not os.access(script_path, os.X_OK):
            try:
                os.chmod(script_path, 0o755)
                logger.info(f"Script {script_path} rendu exécutable")
            except Exception as e:
                logger.error(f"Impossible de rendre le script {script_path} exécutable: {str(e)}")
                return False
        
        return True
    
    def run_learning_optimization(self, phase: str = "training", sample_size: int = 10) -> Dict[str, Any]:
        """
        Exécute l'optimisation du taux d'apprentissage
        
        Args:
            phase: Phase des puzzles
            sample_size: Taille de l'échantillon
            
        Returns:
            Résultats de l'optimisation
        """
        logger.info(f"Exécution de l'optimisation du taux d'apprentissage (phase: {phase}, échantillon: {sample_size})")
        
        script_path = "optimize_learning_rate.py"
        if not self.ensure_script_executable(script_path):
            return {"error": "Script d'optimisation non disponible"}
        
        # Construire la commande
        cmd = [
            sys.executable,
            script_path,
            "--phase", phase,
            "--sample", str(sample_size),
            "--max-epochs", str(self.max_epochs)
        ]
        
        if self.use_gpu:
            cmd.append("--gpu")
        
        # Exécuter la commande
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Erreur lors de l'optimisation: {result.stderr}")
                return {"error": "Erreur d'optimisation", "details": result.stderr}
            
            # Chercher le répertoire de résultats le plus récent
            lr_dirs = [d for d in os.listdir(".") if d.startswith("lr_optimization_")]
            if not lr_dirs:
                logger.error("Aucun répertoire de résultats d'optimisation trouvé")
                return {"error": "Résultats d'optimisation non trouvés"}
            
            # Trier par date de modification (le plus récent en premier)
            lr_dirs.sort(key=lambda d: os.path.getmtime(d), reverse=True)
            
            # Charger le résumé
            summary_path = os.path.join(lr_dirs[0], f"{phase}_optimization_summary.json")
            if not os.path.exists(summary_path):
                logger.error(f"Résumé d'optimisation non trouvé: {summary_path}")
                return {"error": "Résumé d'optimisation non trouvé"}
            
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            # Extraire le taux d'apprentissage optimal
            optimal_lr = summary.get("average_best_learning_rate", self.learning_rate)
            
            logger.info(f"Optimisation terminée - Taux d'apprentissage optimal: {optimal_lr}")
            
            return {
                "optimal_learning_rate": optimal_lr,
                "summary": summary
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'optimisation: {str(e)}")
            return {"error": str(e)}
    
    def run_training_analysis(self, phase: str = "training", sample_size: int = 10, learning_rate: float = None) -> Dict[str, Any]:
        """
        Exécute l'analyse de l'apprentissage
        
        Args:
            phase: Phase des puzzles
            sample_size: Taille de l'échantillon
            learning_rate: Taux d'apprentissage (utilise le taux par défaut si None)
            
        Returns:
            Résultats de l'analyse
        """
        logger.info(f"Exécution de l'analyse d'apprentissage (phase: {phase}, échantillon: {sample_size}, LR: {learning_rate or self.learning_rate})")
        
        script_path = "run_learning_analysis.py"
        if not self.ensure_script_executable(script_path):
            return {"error": "Script d'analyse non disponible"}
        
        # Construire la commande
        cmd = [
            sys.executable,
            script_path,
            "--phase", phase,
            "--sample", str(sample_size),
            "--max-epochs", str(self.max_epochs),
            "--learning-rate", str(learning_rate or self.learning_rate)
        ]
        
        if self.use_gpu:
            cmd.append("--gpu")
        
        # Exécuter la commande
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Erreur lors de l'analyse: {result.stderr}")
                return {"error": "Erreur d'analyse", "details": result.stderr}
            
            # Chercher le répertoire de résultats le plus récent
            learning_dirs = [d for d in os.listdir(".") if d.startswith("learning_results_")]
            if not learning_dirs:
                logger.error("Aucun répertoire de résultats d'analyse trouvé")
                return {"error": "Résultats d'analyse non trouvés"}
            
            # Trier par date de modification (le plus récent en premier)
            learning_dirs.sort(key=lambda d: os.path.getmtime(d), reverse=True)
            
            # Charger le résumé
            summary_path = os.path.join(learning_dirs[0], f"{phase}_summary.json")
            if not os.path.exists(summary_path):
                logger.error(f"Résumé d'analyse non trouvé: {summary_path}")
                return {"error": "Résumé d'analyse non trouvé"}
            
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            logger.info(f"Analyse terminée - {summary.get('total_puzzles', 0)} puzzles analysés")
            
            return {
                "summary": summary,
                "result_dir": learning_dirs[0]
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse: {str(e)}")
            return {"error": str(e)}
    
    def run_validation(self, 
                     training_puzzles: int = 50, 
                     evaluation_puzzles: int = 10, 
                     test_puzzles: int = 10) -> Dict[str, Any]:
        """
        Exécute la validation à grande échelle
        
        Args:
            training_puzzles: Nombre de puzzles d'entraînement
            evaluation_puzzles: Nombre de puzzles d'évaluation
            test_puzzles: Nombre de puzzles de test
            
        Returns:
            Résultats de la validation
        """
        logger.info(f"Exécution de la validation à grande échelle (training: {training_puzzles}, evaluation: {evaluation_puzzles}, test: {test_puzzles})")
        
        script_path = "run_arc_validation.py"
        if not self.ensure_script_executable(script_path):
            return {"error": "Script de validation non disponible"}
        
        # Construire la commande
        cmd = [
            sys.executable,
            script_path,
            "--training", str(training_puzzles),
            "--evaluation", str(evaluation_puzzles),
            "--test", str(test_puzzles),
            "--batch-size", "10"
        ]
        
        # Exécuter la commande
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Erreur lors de la validation: {result.stderr}")
                return {"error": "Erreur de validation", "details": result.stderr}
            
            # Chercher le répertoire de résultats le plus récent
            validation_dirs = [d for d in os.listdir(".") if d.startswith("validation_results_")]
            if not validation_dirs:
                logger.error("Aucun répertoire de résultats de validation trouvé")
                return {"error": "Résultats de validation non trouvés"}
            
            # Trier par date de modification (le plus récent en premier)
            validation_dirs.sort(key=lambda d: os.path.getmtime(d), reverse=True)
            
            # Charger le résumé global
            summary_path = os.path.join(validation_dirs[0], "global_stats.json")
            if not os.path.exists(summary_path):
                logger.error(f"Résumé de validation non trouvé: {summary_path}")
                return {"error": "Résumé de validation non trouvé"}
            
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            logger.info(f"Validation terminée - {summary.get('total_puzzles', 0)} puzzles validés")
            
            return {
                "summary": summary,
                "result_dir": validation_dirs[0]
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la validation: {str(e)}")
            return {"error": str(e)}
    
    def prepare_kaggle_submission(self) -> Dict[str, Any]:
        """
        Prépare la soumission pour Kaggle
        
        Returns:
            Résultat de la préparation
        """
        logger.info("Préparation de la soumission pour Kaggle")
        
        script_path = "kaggle_neurax_integration.py"
        if not self.ensure_script_executable(script_path):
            return {"error": "Script d'intégration Kaggle non disponible"}
        
        # Configurer l'API Kaggle
        kaggle_dir = os.path.expanduser("~/.kaggle")
        os.makedirs(kaggle_dir, exist_ok=True)
        
        # Créer le fichier de configuration
        kaggle_config = {
            "username": self.kaggle_user,
            "key": self.kaggle_key
        }
        
        with open(os.path.join(kaggle_dir, "kaggle.json"), "w") as f:
            json.dump(kaggle_config, f)
        
        # Restreindre les permissions
        try:
            os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)
        except Exception as e:
            logger.warning(f"Impossible de modifier les permissions du fichier de configuration: {str(e)}")
        
        # Construire la commande
        cmd = [
            sys.executable,
            script_path,
            "--download-only"
        ]
        
        # Exécuter la commande
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Erreur lors de la préparation: {result.stderr}")
                return {"error": "Erreur de préparation", "details": result.stderr}
            
            logger.info("Préparation terminée pour la soumission Kaggle")
            
            return {
                "status": "success",
                "kaggle_user": self.kaggle_user
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la préparation: {str(e)}")
            return {"error": str(e)}
    
    def generate_final_report(self, results: Dict[str, Any]) -> None:
        """
        Génère le rapport final de la chaîne de traitement
        
        Args:
            results: Résultats des différentes étapes
        """
        logger.info("Génération du rapport final")
        
        # Créer le contenu du rapport
        report = f"""# Rapport Final de la Chaîne de Traitement Neurax2

## Résumé Exécutif

Ce rapport présente les résultats de l'exécution complète de la chaîne de traitement Neurax2 pour les puzzles ARC.

## Configuration

- **GPU**: {"Activé" if self.use_gpu else "Désactivé"}
- **Optimisation du taux d'apprentissage**: {"Activée" if self.optimize_learning else "Désactivée"}
- **Taux d'apprentissage**: {results.get("optimal_learning_rate", self.learning_rate)}
- **Nombre maximum d'epochs**: {self.max_epochs}
- **API Kaggle**: Configurée pour l'utilisateur {self.kaggle_user}

## Optimisation du Taux d'Apprentissage

"""
        
        # Ajouter les résultats de l'optimisation
        if "optimization_results" in results and "error" not in results["optimization_results"]:
            opt_summary = results["optimization_results"].get("summary", {})
            report += f"""L'optimisation a été réalisée sur {opt_summary.get("total_puzzles", 0)} puzzles et a donné un taux d'apprentissage optimal de {results.get("optimal_learning_rate", self.learning_rate):.6f}.

**Distribution des taux optimaux**:
"""
            
            # Ajouter la distribution des taux d'apprentissage
            for lr, count in opt_summary.get("learning_rate_distribution", {}).items():
                report += f"- **LR = {lr}**: {count} puzzles\n"
        else:
            report += f"Aucune optimisation n'a été effectuée. Utilisation du taux d'apprentissage par défaut: {self.learning_rate}.\n"
        
        report += """
## Analyse de l'Apprentissage

"""
        
        # Ajouter les résultats de l'analyse d'apprentissage
        if "training_results" in results and "error" not in results["training_results"]:
            train_summary = results["training_results"].get("summary", {})
            report += f"""L'analyse de l'apprentissage a été réalisée sur {train_summary.get("total_puzzles", 0)} puzzles et a donné les résultats suivants:

- **Taux de convergence**: {train_summary.get("converged_rate", 0):.1f}%
- **Taux de réussite de traitement**: {train_summary.get("processing_success_rate", 0):.1f}%
- **Nombre moyen d'epochs**: {train_summary.get("average_epochs", 0):.1f}
- **Perte moyenne finale**: {train_summary.get("average_final_loss", 0):.6f}

**Distribution des epochs**:
"""
            
            # Ajouter la distribution des epochs
            for bin_label, count in train_summary.get("epoch_distribution", {}).items():
                report += f"- **{bin_label}**: {count} puzzles\n"
        else:
            report += "Aucune analyse d'apprentissage n'a été effectuée.\n"
        
        report += """
## Validation à Grande Échelle

"""
        
        # Ajouter les résultats de la validation
        if "validation_results" in results and "error" not in results["validation_results"]:
            val_summary = results["validation_results"].get("summary", {})
            report += f"""La validation a été réalisée sur {val_summary.get("total_puzzles", 0)} puzzles et a donné un taux de réussite global de {val_summary.get("overall_success_rate", 0):.1f}%.

**Résultats par phase**:

| Phase | Puzzles | Réussis | Taux de Réussite |
|-------|---------|---------|------------------|
"""
            
            # Ajouter les résultats par phase
            for phase, phase_stats in val_summary.get("phases", {}).items():
                report += f"| {phase.capitalize()} | {phase_stats.get('total', 0)} | {phase_stats.get('success', 0)} | {phase_stats.get('success_rate', 0):.1f}% |\n"
        else:
            report += "Aucune validation n'a été effectuée.\n"
        
        report += """
## Préparation pour Kaggle

"""
        
        # Ajouter les résultats de la préparation Kaggle
        if "kaggle_results" in results and "error" not in results["kaggle_results"]:
            report += f"""La préparation pour la soumission Kaggle a été réalisée avec succès pour l'utilisateur {self.kaggle_user}.

Les données ont été téléchargées et organisées dans la structure requise pour la soumission.
"""
        else:
            report += "La préparation pour Kaggle n'a pas été effectuée ou a échoué.\n"
        
        report += f"""
## Conclusion

La chaîne de traitement Neurax2 a été exécutée {"avec succès" if "error" not in results else "partiellement"} et a montré {"d'excellents" if val_summary.get("overall_success_rate", 0) > 95 else "de bons" if val_summary.get("overall_success_rate", 0) > 80 else "des"} résultats.

Le système est {"prêt" if "error" not in results else "partiellement prêt"} pour la soumission à la compétition ARC-Prize-2025 sur Kaggle.

---

*Rapport généré le {datetime.now().strftime('%Y-%m-%d à %H:%M:%S')}*
"""
        
        # Enregistrer le rapport
        report_path = os.path.join(self.output_dir, "neurax_pipeline_report.md")
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Rapport final généré: {report_path}")
    
    def run_full_pipeline(self, 
                       optimization_sample: int = 10,
                       training_sample: int = 10,
                       validation_training: int = 50,
                       validation_evaluation: int = 10,
                       validation_test: int = 10) -> Dict[str, Any]:
        """
        Exécute la chaîne complète de traitement
        
        Args:
            optimization_sample: Taille de l'échantillon pour l'optimisation
            training_sample: Taille de l'échantillon pour l'analyse d'apprentissage
            validation_training: Nombre de puzzles d'entraînement pour la validation
            validation_evaluation: Nombre de puzzles d'évaluation pour la validation
            validation_test: Nombre de puzzles de test pour la validation
            
        Returns:
            Résultats de la chaîne de traitement
        """
        logger.info("Démarrage de la chaîne complète de traitement")
        
        results = {}
        
        # Étape 1: Optimisation du taux d'apprentissage
        if self.optimize_learning:
            logger.info("Étape 1: Optimisation du taux d'apprentissage")
            optimization_results = self.run_learning_optimization("training", optimization_sample)
            results["optimization_results"] = optimization_results
            
            if "error" in optimization_results:
                logger.error(f"Erreur lors de l'optimisation: {optimization_results['error']}")
            else:
                self.learning_rate = optimization_results.get("optimal_learning_rate", self.learning_rate)
                results["optimal_learning_rate"] = self.learning_rate
                logger.info(f"Taux d'apprentissage optimal trouvé: {self.learning_rate}")
        else:
            logger.info("Optimisation du taux d'apprentissage désactivée")
            results["optimal_learning_rate"] = self.learning_rate
        
        # Étape 2: Analyse de l'apprentissage
        logger.info("Étape 2: Analyse de l'apprentissage")
        training_results = self.run_training_analysis("training", training_sample, self.learning_rate)
        results["training_results"] = training_results
        
        if "error" in training_results:
            logger.error(f"Erreur lors de l'analyse d'apprentissage: {training_results['error']}")
        
        # Étape 3: Validation à grande échelle
        logger.info("Étape 3: Validation à grande échelle")
        validation_results = self.run_validation(validation_training, validation_evaluation, validation_test)
        results["validation_results"] = validation_results
        
        if "error" in validation_results:
            logger.error(f"Erreur lors de la validation: {validation_results['error']}")
        
        # Étape 4: Préparation pour Kaggle
        logger.info("Étape 4: Préparation pour Kaggle")
        kaggle_results = self.prepare_kaggle_submission()
        results["kaggle_results"] = kaggle_results
        
        if "error" in kaggle_results:
            logger.error(f"Erreur lors de la préparation pour Kaggle: {kaggle_results['error']}")
        
        # Étape 5: Génération du rapport final
        logger.info("Étape 5: Génération du rapport final")
        self.generate_final_report(results)
        
        logger.info("Chaîne de traitement terminée")
        
        return results


def main():
    """
    Point d'entrée principal
    """
    parser = argparse.ArgumentParser(description="Chaîne complète de traitement Neurax2")
    parser.add_argument("--output", type=str, default=None,
                      help="Répertoire de sortie")
    parser.add_argument("--gpu", action="store_true",
                      help="Utiliser le GPU si disponible")
    parser.add_argument("--no-optimize", action="store_true",
                      help="Désactiver l'optimisation du taux d'apprentissage")
    parser.add_argument("--learning-rate", type=float, default=0.1,
                      help="Taux d'apprentissage par défaut")
    parser.add_argument("--max-epochs", type=int, default=50,
                      help="Nombre maximum d'epochs")
    parser.add_argument("--kaggle-user", type=str, default="ndarray2000",
                      help="Nom d'utilisateur Kaggle")
    parser.add_argument("--kaggle-key", type=str, default="5354ea3f21950428c738b880332b0a5e",
                      help="Clé API Kaggle")
    parser.add_argument("--opt-sample", type=int, default=10,
                      help="Taille de l'échantillon pour l'optimisation")
    parser.add_argument("--train-sample", type=int, default=10,
                      help="Taille de l'échantillon pour l'analyse d'apprentissage")
    parser.add_argument("--val-training", type=int, default=50,
                      help="Nombre de puzzles d'entraînement pour la validation")
    parser.add_argument("--val-evaluation", type=int, default=10,
                      help="Nombre de puzzles d'évaluation pour la validation")
    parser.add_argument("--val-test", type=int, default=10,
                      help="Nombre de puzzles de test pour la validation")
    
    args = parser.parse_args()
    
    # Créer la chaîne de traitement
    pipeline = NeuraxPipeline(
        output_dir=args.output,
        use_gpu=args.gpu,
        optimize_learning=not args.no_optimize,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        kaggle_user=args.kaggle_user,
        kaggle_key=args.kaggle_key
    )
    
    # Exécuter la chaîne complète
    pipeline.run_full_pipeline(
        optimization_sample=args.opt_sample,
        training_sample=args.train_sample,
        validation_training=args.val_training,
        validation_evaluation=args.val_evaluation,
        validation_test=args.val_test
    )


if __name__ == "__main__":
    main()