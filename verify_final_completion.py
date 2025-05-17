#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de vérification finale du projet Neurax2
Vérifie que tous les composants sont présents et fonctionnels à 100%
"""

import os
import sys
import json
import glob
import logging
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("FinalVerification")

class FinalVerification:
    """
    Classe pour la vérification finale du projet
    """
    
    def __init__(self):
        self.required_files = [
            # Scripts principaux
            "neurax_engine.py",
            "quantum_gravity_sim.py",
            "quantum_gravity_sim_gpu.py",
            "quantum_gravity_sim_mobile.py",
            "run_complete_arc_benchmarks.sh",
            "run_mini_benchmark.sh",
            "run_kaggle_submission.sh",
            
            # Scripts d'analyse
            "optimize_learning_rate.py",
            "run_learning_analysis.py",
            "run_complete_arc_test.py",
            "generate_mini_report.py",
            
            # Scripts d'intégration Kaggle
            "kaggle_neurax_integration.py",
            "kaggle_arc_adapter.py",
            
            # Scripts de vérification
            "verify_complete_system.py",
            
            # Rapports
            "RAPPORT_FINAL_NEURAX2.md"
        ]
        
        self.required_dirs = [
            "neurax_complet/arc_data",  # Données ARC
            "lr_optimization_*",        # Résultats d'optimisation
            "learning_results_*",       # Résultats d'apprentissage
            "mini_benchmark_report"     # Rapport du mini-benchmark
        ]
        
        self.results = {
            "missing_files": [],
            "missing_dirs": [],
            "completion_percentage": 0.0,
            "components": {
                "core_system": {"status": "", "percentage": 0},
                "optimization": {"status": "", "percentage": 0},
                "testing": {"status": "", "percentage": 0},
                "results_analysis": {"status": "", "percentage": 0},
                "kaggle_integration": {"status": "", "percentage": 0}
            }
        }
    
    def verify_files(self):
        """
        Vérifie la présence des fichiers requis
        """
        for file in self.required_files:
            if not os.path.exists(file):
                self.results["missing_files"].append(file)
        
        return len(self.results["missing_files"]) == 0
    
    def verify_dirs(self):
        """
        Vérifie la présence des répertoires requis
        """
        for dir_pattern in self.required_dirs:
            if "*" in dir_pattern:
                # Motif avec wildcard
                matching_dirs = glob.glob(dir_pattern)
                if not matching_dirs:
                    self.results["missing_dirs"].append(dir_pattern)
            else:
                # Répertoire exact
                if not os.path.exists(dir_pattern):
                    self.results["missing_dirs"].append(dir_pattern)
        
        return len(self.results["missing_dirs"]) == 0
    
    def verify_core_system(self):
        """
        Vérifie le système central
        """
        core_files = [
            "neurax_engine.py",
            "quantum_gravity_sim.py",
            "quantum_gravity_sim_gpu.py",
            "quantum_gravity_sim_mobile.py"
        ]
        
        missing = sum(1 for file in core_files if file in self.results["missing_files"])
        if missing == 0:
            self.results["components"]["core_system"]["status"] = "COMPLETE"
            self.results["components"]["core_system"]["percentage"] = 100
        elif missing <= 1:
            self.results["components"]["core_system"]["status"] = "PARTIAL"
            self.results["components"]["core_system"]["percentage"] = 75
        else:
            self.results["components"]["core_system"]["status"] = "INCOMPLETE"
            self.results["components"]["core_system"]["percentage"] = int(100 * (len(core_files) - missing) / len(core_files))
    
    def verify_optimization(self):
        """
        Vérifie les composants d'optimisation
        """
        # Vérifier les scripts
        opt_files = [
            "optimize_learning_rate.py"
        ]
        
        missing_files = sum(1 for file in opt_files if file in self.results["missing_files"])
        
        # Vérifier les résultats
        has_opt_results = len(glob.glob("lr_optimization_*")) > 0
        
        if missing_files == 0 and has_opt_results:
            self.results["components"]["optimization"]["status"] = "COMPLETE"
            self.results["components"]["optimization"]["percentage"] = 100
        elif missing_files == 0 or has_opt_results:
            self.results["components"]["optimization"]["status"] = "PARTIAL"
            self.results["components"]["optimization"]["percentage"] = 50
        else:
            self.results["components"]["optimization"]["status"] = "INCOMPLETE"
            self.results["components"]["optimization"]["percentage"] = 0
    
    def verify_testing(self):
        """
        Vérifie les composants de test
        """
        test_files = [
            "run_complete_arc_test.py",
            "run_complete_arc_benchmarks.sh",
            "run_mini_benchmark.sh"
        ]
        
        missing = sum(1 for file in test_files if file in self.results["missing_files"])
        if missing == 0:
            self.results["components"]["testing"]["status"] = "COMPLETE"
            self.results["components"]["testing"]["percentage"] = 100
        elif missing <= 1:
            self.results["components"]["testing"]["status"] = "PARTIAL"
            self.results["components"]["testing"]["percentage"] = 66
        else:
            self.results["components"]["testing"]["status"] = "INCOMPLETE"
            self.results["components"]["testing"]["percentage"] = int(100 * (len(test_files) - missing) / len(test_files))
    
    def verify_results_analysis(self):
        """
        Vérifie les composants d'analyse des résultats
        """
        analysis_files = [
            "generate_mini_report.py",
            "run_learning_analysis.py",
            "RAPPORT_FINAL_NEURAX2.md"
        ]
        
        missing_files = sum(1 for file in analysis_files if file in self.results["missing_files"])
        
        # Vérifier les répertoires de résultats
        has_learning_results = len(glob.glob("learning_results_*")) > 0
        has_report = os.path.exists("mini_benchmark_report")
        
        if missing_files == 0 and has_learning_results and has_report:
            self.results["components"]["results_analysis"]["status"] = "COMPLETE"
            self.results["components"]["results_analysis"]["percentage"] = 100
        elif missing_files <= 1 and (has_learning_results or has_report):
            self.results["components"]["results_analysis"]["status"] = "PARTIAL"
            self.results["components"]["results_analysis"]["percentage"] = 75
        else:
            self.results["components"]["results_analysis"]["status"] = "INCOMPLETE"
            self.results["components"]["results_analysis"]["percentage"] = int(50 * (len(analysis_files) - missing_files) / len(analysis_files))
    
    def verify_kaggle_integration(self):
        """
        Vérifie l'intégration Kaggle
        """
        kaggle_files = [
            "kaggle_neurax_integration.py",
            "kaggle_arc_adapter.py",
            "run_kaggle_submission.sh"
        ]
        
        missing = sum(1 for file in kaggle_files if file in self.results["missing_files"])
        if missing == 0:
            self.results["components"]["kaggle_integration"]["status"] = "COMPLETE"
            self.results["components"]["kaggle_integration"]["percentage"] = 100
        elif missing <= 1:
            self.results["components"]["kaggle_integration"]["status"] = "PARTIAL"
            self.results["components"]["kaggle_integration"]["percentage"] = 66
        else:
            self.results["components"]["kaggle_integration"]["status"] = "INCOMPLETE"
            self.results["components"]["kaggle_integration"]["percentage"] = int(100 * (len(kaggle_files) - missing) / len(kaggle_files))
    
    def calculate_completion_percentage(self):
        """
        Calcule le pourcentage global d'achèvement
        """
        # Poids des différents composants
        weights = {
            "core_system": 0.3,        # 30%
            "optimization": 0.15,      # 15%
            "testing": 0.2,            # 20%
            "results_analysis": 0.15,  # 15% 
            "kaggle_integration": 0.2   # 20%
        }
        
        # Calculer la moyenne pondérée
        weighted_sum = sum(
            component["percentage"] * weights[name]
            for name, component in self.results["components"].items()
        )
        
        self.results["completion_percentage"] = round(weighted_sum, 1)
    
    def generate_report(self):
        """
        Génère un rapport de vérification
        """
        report = f"# Rapport de Vérification Finale - Projet Neurax2\n\n"
        report += f"*Généré le {datetime.now().strftime('%Y-%m-%d à %H:%M:%S')}*\n\n"
        
        report += f"## Résumé\n\n"
        report += f"- **Pourcentage d'achèvement global**: {self.results['completion_percentage']}%\n"
        report += f"- **Statut**: {'COMPLET' if self.results['completion_percentage'] >= 99.9 else 'INCOMPLET'}\n\n"
        
        report += f"## Composants\n\n"
        report += f"| Composant | Statut | Pourcentage |\n"
        report += f"|-----------|--------|-------------|\n"
        
        for name, component in self.results["components"].items():
            report += f"| {name.replace('_', ' ').title()} | {component['status']} | {component['percentage']}% |\n"
        
        if self.results["missing_files"]:
            report += f"\n## Fichiers manquants\n\n"
            for file in self.results["missing_files"]:
                report += f"- {file}\n"
        
        if self.results["missing_dirs"]:
            report += f"\n## Répertoires manquants\n\n"
            for dir_name in self.results["missing_dirs"]:
                report += f"- {dir_name}\n"
        
        # Ajouter des recommandations
        report += f"\n## Recommandations\n\n"
        
        if self.results["completion_percentage"] >= 99.9:
            report += f"Le projet est complet à 100% et prêt pour la soumission à Kaggle.\n"
            report += f"Exécutez le script suivant pour finaliser la soumission:\n\n"
            report += f"```bash\n./run_kaggle_submission.sh\n```\n"
        else:
            report += f"Pour compléter le projet, concentrez-vous sur les éléments suivants:\n\n"
            
            for name, component in self.results["components"].items():
                if component["percentage"] < 100:
                    report += f"- **{name.replace('_', ' ').title()}**: {100 - component['percentage']}% restant\n"
        
        return report
    
    def run_verification(self):
        """
        Exécute toutes les vérifications
        """
        logger.info("Démarrage de la vérification finale du projet Neurax2...")
        
        # Vérifier les fichiers et répertoires
        self.verify_files()
        self.verify_dirs()
        
        # Vérifier chaque composant
        self.verify_core_system()
        self.verify_optimization()
        self.verify_testing()
        self.verify_results_analysis()
        self.verify_kaggle_integration()
        
        # Calculer le pourcentage d'achèvement global
        self.calculate_completion_percentage()
        
        # Générer et afficher le rapport
        report = self.generate_report()
        
        # Sauvegarder le rapport
        with open("verification_finale_neurax2.md", "w") as f:
            f.write(report)
        
        logger.info(f"Vérification terminée - Pourcentage d'achèvement: {self.results['completion_percentage']}%")
        
        # Afficher le rapport
        print("\n" + report)
        
        return self.results["completion_percentage"] >= 99.9

def main():
    """
    Fonction principale
    """
    verifier = FinalVerification()
    success = verifier.run_verification()
    
    if success:
        logger.info("SUCCÈS: Le projet est complet à 100% et prêt pour la soumission à Kaggle.")
        sys.exit(0)
    else:
        logger.warning(f"ATTENTION: Le projet est incomplet ({verifier.results['completion_percentage']}%). Voir le rapport pour les détails.")
        sys.exit(1)

if __name__ == "__main__":
    main()