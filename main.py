#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import sys
import argparse
from datetime import datetime

# Ajouter le répertoire courant au chemin pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from comprehensive_test_framework import TestSuite

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"arc_learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("main")

def main():
    # Parser pour les arguments en ligne de commande
    parser = argparse.ArgumentParser(description="Exécution des tests Neurax2 sur les puzzles ARC-Prize-2025")
    parser.add_argument("--training", type=int, default=1000, 
                        help="Nombre de puzzles d'entraînement à traiter (défaut: 1000 = tous)")
    parser.add_argument("--evaluation", type=int, default=120, 
                        help="Nombre de puzzles d'évaluation à traiter (défaut: 120 = tous)")
    parser.add_argument("--test", type=int, default=240, 
                        help="Nombre de puzzles de test à traiter (défaut: 240 = tous)")
    parser.add_argument("--arc-data", type=str, default="./neurax_complet/arc_data",
                        help="Chemin vers les données ARC (défaut: ./neurax_complet/arc_data)")
    parser.add_argument("--report", type=str, default="analyse_resultats_reels.md",
                        help="Nom du fichier de rapport (défaut: analyse_resultats_reels.md)")
    parser.add_argument("--export-prefix", type=str, default="arc",
                        help="Préfixe pour les fichiers d'export (défaut: arc)")
    
    args = parser.parse_args()
    
    logger.info("=== DÉMARRAGE DES TESTS NEURAX2 SUR ARC-PRIZE-2025 ===")
    logger.info(f"Configuration:")
    logger.info(f"- Puzzles d'entraînement: {args.training}")
    logger.info(f"- Puzzles d'évaluation: {args.evaluation}")
    logger.info(f"- Puzzles de test: {args.test}")
    logger.info(f"- Données ARC: {args.arc_data}")
    
    # Créer et exécuter la suite de tests
    test_suite = TestSuite(arc_data_path=args.arc_data)
    results = test_suite.run_all_tests(
        training_puzzles=args.training,
        evaluation_puzzles=args.evaluation,
        test_puzzles=args.test
    )
    
    # Afficher les résultats détaillés d'apprentissage
    summary = results.get_summary()
    logger.info("=== RÉSULTATS APPRENTISSAGE ARC ===")
    logger.info(f"Tests exécutés: {summary.get('total_tests', 0)}")
    logger.info(f"Puzzles traités: {summary.get('puzzle_count', 0)}")
    
    # Détails par phase
    for phase in ["training", "evaluation", "test"]:
        phase_count = summary.get('puzzles_by_phase', {}).get(phase, 0)
        success_count = summary.get('success_by_phase', {}).get(phase, 0)
        if phase_count > 0:
            success_rate = success_count / phase_count * 100
            logger.info(f"Phase {phase.capitalize()}: {success_count}/{phase_count} puzzles réussis ({success_rate:.2f}%)")
    
    # Statistiques globales
    logger.info(f"Taux de réussite global: {summary.get('success_rate', 0)*100:.2f}%")
    logger.info(f"Précision moyenne: {summary.get('average_accuracy', 0)*100:.2f}%")
    
    # Noms des fichiers d'export
    report_file = args.report
    results_csv = f"{args.export_prefix}_tests_results.csv"
    puzzles_csv = f"{args.export_prefix}_puzzles_results.csv"
    results_json = f"{args.export_prefix}_tests_results.json"
    puzzles_json = f"{args.export_prefix}_puzzles_detailed_results.json"
    
    # Générer les exports
    logger.info(f"Génération du rapport détaillé: {report_file}")
    results.generate_detailed_report(report_file)
    
    logger.info(f"Export des résultats au format CSV: {results_csv}, {puzzles_csv}")
    results.export_to_csv(results_csv)
    results.export_arc_results_to_csv(puzzles_csv)
    
    logger.info(f"Export des résultats au format JSON: {results_json}, {puzzles_json}")
    results.export_to_json(results_json)
    results.export_puzzle_results_to_json(puzzles_json)
    
    logger.info("=== TESTS TERMINÉS ===")

if __name__ == "__main__":
    main()