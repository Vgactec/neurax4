#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script pour exécuter l'analyse complète du système Neurax sur les puzzles ARC
"""

import os
import sys
import logging
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("NeuraxExecutor")

def main():
    """
    Point d'entrée principal pour l'exécution de l'analyse
    """
    logger.info("Démarrage de l'analyse du système Neurax")
    
    # Ajouter le chemin du module neurax_complet
    neurax_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "neurax_complet")
    neurax_module = os.path.join(neurax_root, "neurax_complet")
    sys.path.insert(0, neurax_root)
    sys.path.insert(0, neurax_module)
    
    # Exécuter les tests complets à partir du comprehensive_test_framework.py
    try:
        # Import dynamique pour éviter les problèmes de dépendances
        sys.path.insert(0, neurax_module)
        
        # Tenter d'importer le framework de test
        try:
            from comprehensive_test_framework import TestSuite
            logger.info("Framework de test importé avec succès")
        except ImportError as e:
            logger.error(f"Erreur lors de l'importation du framework de test: {e}")
            logger.info("Tentative d'importation alternative...")
            
            # Essayer une autre méthode d'importation
            import importlib.util
            spec = importlib.util.spec_from_file_location("test_framework", 
                                                        os.path.join(neurax_module, "comprehensive_test_framework.py"))
            if spec is None:
                logger.error("Impossible de trouver le module comprehensive_test_framework.py")
                return
                
            test_framework = importlib.util.module_from_spec(spec)
            sys.modules["test_framework"] = test_framework
            spec.loader.exec_module(test_framework)
            TestSuite = test_framework.TestSuite
            logger.info("Framework de test importé avec succès (méthode alternative)")
        
        # Exécuter les tests
        test_suite = TestSuite()
        logger.info("Instance de TestSuite créée, exécution des tests...")
        
        # Exécuter les tests sur tous les puzzles ARC
        # Note: la méthode run_all_tests pourrait ne pas exister exactement sous ce nom
        # On tente plusieurs approches
        try:
            logger.info("Exécution de run_all_tests...")
            results = test_suite.run_all_tests()
        except AttributeError:
            try:
                logger.info("run_all_tests non trouvé, tentative avec test_arc_puzzles_with_neurax...")
                results = test_suite.test_arc_puzzles_with_neurax()
            except AttributeError:
                logger.info("Exécution des tests individuels...")
                # Exécution manuelle des tests importants
                test_suite.test_quantum_gravity_simulator()
                test_suite.test_arc_puzzles_with_neurax()
                results = test_suite.results
        
        logger.info("Tests terminés avec succès")
        return results
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution des tests: {e}")
        return None

if __name__ == "__main__":
    results = main()
    if results:
        logger.info("Analyse terminée avec succès")
    else:
        logger.warning("L'analyse a rencontré des problèmes")