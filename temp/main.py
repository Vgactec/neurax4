#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import sys
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from comprehensive_test_framework import TestSuite

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("arc_learning.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("main")

def main():
    logger.info("Démarrage des tests complets")
    test_suite = TestSuite(arc_data_path="./neurax_complet/arc_data")
    results = test_suite.run_all_tests()
    
    # Afficher les résultats détaillés d'apprentissage
    summary = results.get_summary()
    logger.info("=== RÉSULTATS APPRENTISSAGE ARC ===")
    logger.info(f"Puzzles traités: {summary.get('puzzle_count', 0)}")
    logger.info(f"Taux de réussite: {summary.get('success_rate', 0)*100:.2f}%")
    logger.info(f"Précision moyenne: {summary.get('average_accuracy', 0)*100:.2f}%")
    
    # Générer le rapport détaillé
    results.generate_detailed_report("analyse_resultats_reels.md")
    
    # Exporter les résultats
    results.export_to_csv("arc_tests_results.csv")
    results.export_arc_results_to_csv("arc_puzzles_results.csv")
    logger.info("Tests terminés")

if __name__ == "__main__":
    main()