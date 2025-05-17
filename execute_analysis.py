from comprehensive_test_framework import TestSuite
import logging
import time

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("neurax_validation_tests.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("ValidationTests")

def main():
    logger.info("Démarrage des tests de validation")
    start_time = time.time()

    # Initialisation de la suite de tests
    test_suite = TestSuite()

    # Exécution des tests
    results = test_suite.run_all_tests(
        training_puzzles=400,  # Nombre de puzzles d'entraînement à tester
        evaluation_puzzles=100, # Nombre de puzzles d'évaluation
        test_puzzles=200       # Nombre de puzzles de test
    )

    # Export des résultats
    results.export_to_json("validation_results.json")
    results.export_to_csv("validation_results.csv")
    results.generate_detailed_report("validation_analysis.md")

    duration = time.time() - start_time
    logger.info(f"Tests de validation terminés en {duration:.2f} secondes")

if __name__ == "__main__":
    main()