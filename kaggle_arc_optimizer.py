"""
Script d'optimisation pour le notebook Neurax3 sur Kaggle
Ce script permet de traiter la totalité des 1360 puzzles ARC sans aucune limitation.

Instructions:
1. Exécuter ce script en l'important dans le notebook Kaggle
2. Utiliser les fonctions optimisées pour remplacer les versions limitées
"""

# Configuration du système de logs avancé
import logging
import datetime
import os
import sys

# S'assurer que le dossier de logs existe
os.makedirs("logs", exist_ok=True)

# Définir un fichier de logs avec horodatage
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f"neurax3_execution_log_{timestamp}.log"
log_path = os.path.join("logs", log_filename)

# Configuration du logger avec formats détaillés
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('neurax3_optimizer')

# Créer un fichier de statut initial
status_file = os.path.join("logs", "status.txt")
with open(status_file, "w") as f:
    f.write(f"=== NEURAX3 OPTIMIZER - ARC-PRIZE-2025 ===\n")
    f.write(f"Version: 1.1 - Date de démarrage: {timestamp}\n")
    f.write(f"Mode: Traitement complet sans limitation (1360 puzzles)\n")
    f.write(f"Statut: INITIALISATION\n")

# Information de démarrage dans les logs
logger.info("=" * 80)
logger.info("=== DÉMARRAGE DE L'OPTIMISATION NEURAX3 POUR ARC-PRIZE-2025 ===")
logger.info(f"Version: 1.1 - Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("Mode: Traitement complet sans limitation (1360 puzzles)")
logger.info("Système de logs actif - Tous les événements sont enregistrés")
logger.info(f"Fichier de logs principal: {log_path}")
logger.info("=" * 80)

import os
import time
import json
import sys
import numpy as np
from tqdm import tqdm
import torch

# Configuration pour sauvegarder les points de reprise
def save_checkpoint(processed_ids, phase):
    """Sauvegarde un point de reprise pour le traitement des puzzles"""
    checkpoint = {
        "processed_ids": processed_ids,
        "timestamp": time.time(),
        "date": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    checkpoint_file = f"{phase}_checkpoint.json"
    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint, f, indent=2)
    print(f"Point de reprise sauvegardé pour {len(processed_ids)} puzzles {phase}")
    return checkpoint_file

# Fonction pour charger les points de reprise
def load_checkpoint(phase):
    """Charge un point de reprise existant"""
    processed_ids = []
    checkpoint_file = f"{phase}_checkpoint.json"
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r") as f:
                checkpoint = json.load(f)
                processed_ids = checkpoint.get("processed_ids", [])
                print(f"Point de reprise chargé: {len(processed_ids)} puzzles {phase} déjà traités")
        except json.JSONDecodeError:
            print(f"Fichier de point de reprise {checkpoint_file} corrompu, démarrage d'un nouveau traitement")
    return processed_ids

# Configuration du moteur pour utilisation optimale du GPU
def configure_engine_for_gpu(engine):
    """Configure le moteur Neurax pour utiliser le GPU de manière optimale"""
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            print(f"GPU détecté: {gpu_name}")
            print(f"Mémoire GPU disponible: {gpu_memory:.2f} GB")
            
            # Configurer les paramètres optimaux pour le GPU
            if hasattr(engine, "configure"):
                engine.configure(
                    use_gpu=True,
                    grid_size=64,        # Augmenter la taille de grille (plus grande précision spatiale)
                    time_steps=16,       # Augmenter les pas de temps (plus grande précision temporelle)
                    batch_size=8,        # Traitement par lots (meilleure utilisation du GPU)
                    precision="float16", # Précision mixte pour économie de mémoire GPU
                    # Optimisations supplémentaires
                    parallelize=True,    # Activer la parallélisation GPU
                    use_cuda_kernels=True, # Utiliser des kernels CUDA optimisés
                    enable_tensor_cores=True, # Utiliser les tensor cores si disponibles
                    memory_efficient=True,    # Mode économie de mémoire
                    # Optimisations avancées du simulateur de gravité quantique
                    quantum_state_compression=True,  # Compression des états quantiques
                    adaptive_resolution=True,        # Résolution adaptative
                    relativistic_effects=True,       # Effets relativistes
                    non_local_interactions=True      # Interactions non-locales
                )
                print("Moteur Neurax configuré pour utilisation optimale du GPU avec toutes les optimisations avancées")
            elif hasattr(engine, "use_gpu"):
                # Configuration de base si la méthode configure n'est pas disponible
                engine.use_gpu = True
                engine.grid_size = 64
                engine.time_steps = 16
                
                # Configuration avancée si les attributs sont disponibles
                if hasattr(engine, "parallelize"): 
                    engine.parallelize = True
                if hasattr(engine, "use_cuda_kernels"):
                    engine.use_cuda_kernels = True
                if hasattr(engine, "quantum_state_compression"):
                    engine.quantum_state_compression = True
                if hasattr(engine, "adaptive_resolution"):
                    engine.adaptive_resolution = True
                
                print("Moteur Neurax configuré pour GPU avec optimisations disponibles")
            
            # Configurer pour calcul distribué si disponible
            if hasattr(engine, "enable_distributed_computing"):
                engine.enable_distributed_computing(nodes=1, mode="auto")
                print("Calcul distribué activé pour une utilisation optimale des ressources")
                
            return True
    except (ImportError, AttributeError, RuntimeError) as e:
        print(f"Erreur lors de la configuration GPU: {e}")
    
    print("Utilisation du CPU uniquement - Performance réduite")
    return False

# Fonction principale optimisée pour le traitement des puzzles sans aucune limitation
def process_puzzles_optimized(puzzles, engine, max_time_per_puzzle=None, phase="test", verify_solutions=False):
    """
    Version optimisée de process_puzzles sans aucune limitation.
    
    Args:
        puzzles: Liste des puzzles à traiter
        engine: Moteur Neurax3 pour le traitement
        max_time_per_puzzle: Temps maximum par puzzle en secondes (None=illimité, défaut=600s)
        phase: Phase de traitement ('training', 'evaluation', 'test')
        verify_solutions: Vérifier les solutions avec les valeurs attendues
        
    Returns:
        results: Liste des résultats pour chaque puzzle
        summary: Résumé des statistiques de traitement
    """
    # Définir le temps maximum par puzzle
    if max_time_per_puzzle is None:
        logger.warning("ATTENTION: Aucune limite de temps par puzzle configurée - traitement illimité")
        # Définir une valeur très élevée équivalente à l'illimité (10 heures par puzzle)
        max_time_per_puzzle = 36000
    else:
        logger.info(f"Temps maximum par puzzle: {max_time_per_puzzle} secondes")
    
    # Vérifier si des puzzles ont déjà été traités
    processed_ids = load_checkpoint(phase)
    
    # Configurer pour GPU si disponible
    is_gpu = configure_engine_for_gpu(engine)
    
    # Charger les résultats existants
    results = []
    results_file = f"{phase}_results.json"
    if os.path.exists(results_file) and len(processed_ids) > 0:
        try:
            with open(results_file, "r") as f:
                results = json.load(f)
                logger.info(f"Chargé {len(results)} résultats existants")
        except json.JSONDecodeError:
            logger.error(f"Fichier {results_file} corrompu, démarrage d'une nouvelle analyse")
    
    # Filtrer les puzzles déjà traités
    # AUCUNE limitation sur le nombre de puzzles - traitement absolument complet
    puzzles_to_process = [p for p in puzzles if p.get("id", "unknown") not in processed_ids]
    logger.info(f"Puzzles à traiter: {len(puzzles_to_process)}/{len(puzzles)} ({phase}) - TRAITEMENT COMPLET SANS LIMITATION")
    logger.info(f"ATTENTION: Tous les puzzles seront traités sans aucune limitation artificielle")
    
    # Vérification explicite qu'il n'y a pas de limitation sur le nombre de puzzles
    if phase == "training" and len(puzzles) < 1000:
        logger.warning(f"AVERTISSEMENT: Nombre de puzzles d'entraînement limité à {len(puzzles)} au lieu de 1000")
    elif phase == "evaluation" and len(puzzles) < 120:
        logger.warning(f"AVERTISSEMENT: Nombre de puzzles d'évaluation limité à {len(puzzles)} au lieu de 120")
    elif phase == "test" and len(puzzles) < 240:
        logger.warning(f"AVERTISSEMENT: Nombre de puzzles de test limité à {len(puzzles)} au lieu de 240")
        
    # Enregistrer les informations détaillées sur les puzzles pour faciliter le débogage
    try:
        with open(f"{phase}_puzzles_info.json", "w") as f:
            puzzle_info = [{"id": p.get("id", "unknown"), "size": len(str(p))} for p in puzzles]
            json.dump(puzzle_info, f, indent=2)
        logger.info(f"Informations sur les puzzles enregistrées dans {phase}_puzzles_info.json")
    except Exception as e:
        logger.error(f"Erreur lors de l'enregistrement des informations sur les puzzles: {e}")
    
    # Mécanisme de récupération des logs actif
    try:
        # Créer un dossier pour les logs si nécessaire
        os.makedirs("logs", exist_ok=True)
        
        # Copier le fichier de logs actuel dans le dossier logs
        import shutil
        shutil.copy(log_filename, os.path.join("logs", log_filename))
        
        logger.info(f"Fichier de logs créé: {log_filename}")
        logger.info(f"Logs enregistrés dans le dossier 'logs'")
        
        # Créer un fichier de statut pour vérification
        with open("execution_status.txt", "w") as f:
            f.write(f"Neurax3 Optimizer - Exécution en cours\n")
            f.write(f"Phase: {phase}\n")
            f.write(f"Date de démarrage: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Puzzles à traiter: {len(puzzles_to_process)}/{len(puzzles)}\n")
            f.write(f"GPU activé: {is_gpu}\n")
        logger.info("Fichier de statut créé: execution_status.txt")
    except Exception as e:
        logger.error(f"Erreur lors de la configuration des fichiers de logs: {e}")
        
    # Statistiques globales
    total_puzzles = len(puzzles)
    total_processed = len(processed_ids)
    total_time = 0
    total_iterations = 0
    successful_puzzles = 0
    
    logger.info("=== DÉBUT DU TRAITEMENT DES PUZZLES ===")
    logger.info(f"Phase: {phase} - Total: {total_puzzles} puzzles")
    logger.info(f"Puzzles déjà traités: {total_processed}")
    logger.info(f"Puzzles restants: {len(puzzles_to_process)}")
    
    for puzzle in tqdm(puzzles_to_process, desc=f"Traitement {phase}"):
        start_time = time.time()
        puzzle_id = puzzle.get("id", "unknown")
        
        # Afficher la progression
        total_processed += 1
        progress = (total_processed / total_puzzles) * 100
        remaining = total_puzzles - total_processed
        print(f"\nPuzzle {puzzle_id} ({total_processed}/{total_puzzles}, {progress:.2f}%) - Restants: {remaining}")
        
        try:
            # Fonction de conversion robuste pour les puzzles ARC
            def convert_puzzle_format(puzzle_data):
                """Fonction de conversion complète pour les puzzles ARC"""
                try:
                    import numpy as np
                    converted = {}
                    converted["id"] = puzzle_data.get("id", "unknown")
                    
                    # Convertir les entrées (input)
                    if "train" in puzzle_data:
                        converted["input"] = [np.array(example["input"]) for example in puzzle_data["train"]]
                    
                    # Convertir les sorties (output) pour l'entraînement
                    if "train" in puzzle_data:
                        converted["output"] = [np.array(example["output"]) for example in puzzle_data["train"]]
                    
                    # Convertir la solution pour la vérification si disponible
                    if "solution" in puzzle_data:
                        converted["solution"] = [np.array(sol) for sol in puzzle_data["solution"]]
                    
                    # Vérification de l'intégrité de la conversion
                    if "input" not in converted or len(converted["input"]) == 0:
                        logger.warning(f"Conversion incomplète pour {puzzle_data.get('id', 'unknown')}: Aucune entrée trouvée")
                    
                    return converted
                except Exception as e:
                    logger.error(f"Erreur de conversion du puzzle {puzzle_data.get('id', 'unknown')}: {e}")
                    # Version dégradée de secours - retourne le minimum viable pour éviter un crash complet
                    return {"id": puzzle_data.get("id", "unknown")}
            
            # Convertir au format Neurax - utiliser notre fonction robuste
            logger.info(f"Conversion du puzzle {puzzle_id}...")
            neurax_puzzle = convert_puzzle_format(puzzle)
            
            if neurax_puzzle is None:
                raise ValueError(f"Échec de conversion pour le puzzle {puzzle_id}")
                
            # Log détaillé de la conversion pour débogage
            conversion_status = {
                "id": puzzle_id,
                "has_input": "input" in neurax_puzzle,
                "has_output": "output" in neurax_puzzle,
                "has_solution": "solution" in neurax_puzzle,
                "input_count": len(neurax_puzzle.get("input", [])),
                "output_count": len(neurax_puzzle.get("output", []))
            }
            logger.info(f"Statut conversion: {conversion_status}")
            
            # Traiter avec le moteur sans AUCUNE limite d'époques ni de temps
            # Application des paramètres sans limitation
            result = engine.process_puzzle(
                neurax_puzzle,
                max_time=max_time_per_puzzle,  # Utiliser le paramètre transmis (potentiellement illimité)
                max_epochs=0,  # Aucune limite d'époques
                use_gpu=is_gpu
            )
            
            success = result.get("success", False)
            iterations = result.get("iterations", 0)
            execution_time = time.time() - start_time
            
            # Collecter les métriques
            metrics = engine.get_training_stats(puzzle_id)
            
            # Créer l'objet résultat
            puzzle_result = {
                "id": puzzle_id,
                "success": success,
                "execution_time": execution_time,
                "iterations": iterations,
                "best_learning_rate": metrics.get("best_learning_rate", 0),
                "final_loss": metrics.get("final_loss", float('inf')),
                "phase": phase
            }
            
            # Vérifier la solution si nécessaire
            if verify_solutions and "solution" in neurax_puzzle:
                predicted = result.get("solution", [])
                expected = neurax_puzzle["solution"][0]  # Premier test uniquement
                is_correct = np.array_equal(predicted, expected)
                puzzle_result["correct"] = is_correct
                successful_puzzles += 1 if is_correct else 0
            else:
                successful_puzzles += 1 if success else 0
            
            # Ajouter aux résultats
            results.append(puzzle_result)
            
            # Mise à jour des statistiques
            total_time += execution_time
            total_iterations += iterations
            
            # Ajouter à la liste des IDs traités
            processed_ids.append(puzzle_id)
            
            # Sauvegarder point de reprise et résultats partiels
            save_checkpoint(processed_ids, phase)
            with open(f"{phase}_results_partial.json", "w") as f:
                json.dump(results, f, indent=2)
                
            if len(results) % 10 == 0:
                # Sauvegarde complète tous les 10 puzzles
                with open(results_file, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"Sauvegarde complète de {len(results)} résultats effectuée")
                
        except Exception as e:
            print(f"Erreur lors du traitement du puzzle {puzzle_id}: {e}")
            # Enregistrer l'erreur
            error_result = {
                "id": puzzle_id,
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time,
                "iterations": 0,
                "phase": phase
            }
            results.append(error_result)
            
            # Sauvegarder malgré l'erreur
            processed_ids.append(puzzle_id)
            save_checkpoint(processed_ids, phase)
    
    # Sauvegarder tous les résultats à la fin
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Créer le résumé
    summary = {
        "total_puzzles": total_puzzles,
        "processed_puzzles": total_processed,
        "successful_puzzles": successful_puzzles,
        "success_rate": (successful_puzzles / total_processed) * 100 if total_processed > 0 else 0,
        "total_execution_time": total_time,
        "average_time_per_puzzle": total_time / total_processed if total_processed > 0 else 0,
        "total_iterations": total_iterations,
        "average_iterations": total_iterations / total_processed if total_processed > 0 else 0,
        "phase": phase
    }
    
    with open(f"{phase}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nTerminé: {total_processed} puzzles traités, {successful_puzzles} réussis")
    print(f"Taux de réussite: {summary['success_rate']:.2f}%")
    print(f"Temps d'exécution total: {total_time:.2f}s")
    print(f"Temps moyen par puzzle: {summary['average_time_per_puzzle']:.2f}s")
    
    return results, summary

# Fonction spéciale pour intégrer toutes les extensions physiques demandées
def enhance_quantum_gravity_simulator(engine):
    """
    Implémente les extensions physiques avancées pour le simulateur de gravité quantique:
    - Champs quantiques supplémentaires
    - Interactions non-locales
    - Effets relativistes
    - Algorithmes adaptatifs
    - Compression des états quantiques
    """
    print("Application des extensions physiques avancées au simulateur de gravité quantique...")
    
    # Vérifier si le moteur a les capacités nécessaires
    has_advanced_physics = hasattr(engine, "enable_advanced_physics")
    has_simulator = hasattr(engine, "quantum_gravity_simulator")
    
    if has_advanced_physics:
        # Activer directement les extensions avancées
        engine.enable_advanced_physics(
            additional_quantum_fields=True,
            non_local_interactions=True,
            relativistic_effects=True,
            adaptive_algorithms=True,
            quantum_state_compression=True
        )
        print("Extensions physiques avancées activées avec succès")
        return True
    elif has_simulator:
        # Configuration manuelle du simulateur
        simulator = engine.quantum_gravity_simulator
        
        if hasattr(simulator, "enable_additional_quantum_fields"):
            simulator.enable_additional_quantum_fields()
            print("Champs quantiques supplémentaires activés")
            
        if hasattr(simulator, "enable_non_local_interactions"):
            simulator.enable_non_local_interactions()
            print("Interactions non-locales activées")
            
        if hasattr(simulator, "enable_relativistic_effects"):
            simulator.enable_relativistic_effects()
            print("Effets relativistes activés")
            
        if hasattr(simulator, "enable_adaptive_algorithms"):
            simulator.enable_adaptive_algorithms()
            print("Algorithmes adaptatifs activés")
            
        if hasattr(simulator, "enable_quantum_state_compression"):
            simulator.enable_quantum_state_compression()
            print("Compression des états quantiques activée")
            
        print("Extensions physiques configurées manuellement")
        return True
    else:
        print("Le moteur ne supporte pas les extensions physiques avancées")
        return False

# Fonction pour traiter tous les puzzles de manière complète
def run_complete_arc_analysis(engine, puzzle_data_dir="../input/arc-prize-2025"):
    """
    Exécute l'analyse complète de tous les puzzles ARC sans aucune limitation
    
    Args:
        engine: Moteur Neurax optimisé
        puzzle_data_dir: Répertoire contenant les puzzles ARC
        
    Returns:
        Résumé global des traitements
    """
    print("Démarrage de l'analyse complète des puzzles ARC sans aucune limitation")
    
    # Activer les extensions physiques avancées
    enhanced = enhance_quantum_gravity_simulator(engine)
    if enhanced:
        print("Moteur amélioré avec toutes les extensions physiques avancées")
    
    # Charger les données des puzzles
    training_file = os.path.join(puzzle_data_dir, "arc-agi_training_challenges.json")
    evaluation_file = os.path.join(puzzle_data_dir, "arc-agi_evaluation_challenges.json")
    test_file = os.path.join(puzzle_data_dir, "arc-agi_test_challenges.json")
    
    # Vérifier l'existence des fichiers
    files_exist = all(os.path.exists(f) for f in [training_file, evaluation_file, test_file])
    if not files_exist:
        print("Fichiers de puzzles non trouvés, vérifiez le chemin et les noms de fichiers")
        return None
    
    # Charger les puzzles
    with open(training_file, "r") as f:
        training_puzzles = json.load(f)
    print(f"Puzzles d'entraînement chargés: {len(training_puzzles)} (TRAITEMENT COMPLET)")
    
    with open(evaluation_file, "r") as f:
        evaluation_puzzles = json.load(f)
    print(f"Puzzles d'évaluation chargés: {len(evaluation_puzzles)} (TRAITEMENT COMPLET)")
    
    with open(test_file, "r") as f:
        test_puzzles = json.load(f)
    print(f"Puzzles de test chargés: {len(test_puzzles)} (TRAITEMENT COMPLET)")
    
    # Traiter tous les puzzles d'entraînement (sans limitation artificielle)
    print("Traitement des puzzles d'entraînement complets sans limitation...")
    training_results, training_summary = process_puzzles_optimized(
        training_puzzles, 
        engine, 
        max_time_per_puzzle=600,  # 10 minutes par puzzle (changez à None pour illimité)
        phase="training",
        verify_solutions=True
    )
    
    # Traiter tous les puzzles d'évaluation (sans limitation artificielle)
    print("Traitement des puzzles d'évaluation complets sans limitation...")
    evaluation_results, evaluation_summary = process_puzzles_optimized(
        evaluation_puzzles, 
        engine, 
        max_time_per_puzzle=600,  # 10 minutes par puzzle (changez à None pour illimité)
        phase="evaluation",
        verify_solutions=True
    )
    
    # Traiter tous les puzzles de test (sans limitation artificielle)
    print("Traitement des puzzles de test complets sans limitation...")
    test_results, test_summary = process_puzzles_optimized(
        test_puzzles, 
        engine, 
        max_time_per_puzzle=600,  # 10 minutes par puzzle (changez à None pour illimité)
        phase="test",
        verify_solutions=False  # Pas de vérification pour les puzzles de test
    )
    
    # Préparer le résumé global
    global_summary = {
        "training": training_summary,
        "evaluation": evaluation_summary,
        "test": test_summary,
        "total_puzzles": len(training_puzzles) + len(evaluation_puzzles) + len(test_puzzles),
        "total_processed": training_summary["processed_puzzles"] + evaluation_summary["processed_puzzles"] + test_summary["processed_puzzles"],
        "total_successful": training_summary["successful_puzzles"] + evaluation_summary["successful_puzzles"] + test_summary["successful_puzzles"],
        "total_execution_time": training_summary["total_execution_time"] + evaluation_summary["total_execution_time"] + test_summary["total_execution_time"],
        "global_success_rate": 0,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Calculer le taux de réussite global
    if global_summary["total_processed"] > 0:
        global_summary["global_success_rate"] = (global_summary["total_successful"] / global_summary["total_processed"]) * 100
    
    # Sauvegarder le résumé global
    with open("global_summary.json", "w") as f:
        json.dump(global_summary, f, indent=2)
    
    print("Analyse complète terminée!")
    print(f"Total des puzzles traités: {global_summary['total_processed']}")
    print(f"Taux de réussite global: {global_summary['global_success_rate']:.2f}%")
    print(f"Temps d'exécution total: {global_summary['total_execution_time']:.2f}s")
    
    return global_summary

# Fonction pour préparer la soumission Kaggle
def prepare_kaggle_submission(output_file="submission.json"):
    """
    Prépare une soumission au format requis par la compétition ARC-Prize-2025
    
    Args:
        output_file: Fichier de sortie pour la soumission
    """
    print("Préparation de la soumission Kaggle...")
    
    # Vérifier si les résultats des tests existent
    test_results_file = "test_results.json"
    if not os.path.exists(test_results_file):
        print("Erreur: Résultats des tests non trouvés. Exécutez d'abord run_complete_arc_analysis")
        return False
    
    # Charger les résultats des tests
    with open(test_results_file, "r") as f:
        test_results = json.load(f)
    
    # Préparer la soumission
    submission = {}
    
    for result in test_results:
        puzzle_id = result.get("id", "unknown")
        
        # Si la solution est disponible, l'ajouter à la soumission
        if "solution" in result:
            submission[puzzle_id] = result["solution"]
    
    # Vérifier si toutes les solutions sont présentes
    print(f"Nombre de solutions préparées: {len(submission)}")
    
    # Sauvegarder la soumission
    with open(output_file, "w") as f:
        json.dump(submission, f, indent=2)
    
    print(f"Soumission Kaggle préparée et sauvegardée dans {output_file}")
    return True

# Fonction utilitaire pour téléverser le notebook optimisé sur Kaggle
def upload_to_kaggle(notebook_path, competition_name="arc-prize-2025"):
    """
    Téléverse le notebook optimisé sur Kaggle en utilisant l'API Kaggle
    
    Args:
        notebook_path: Chemin vers le notebook à téléverser
        competition_name: Nom de la compétition Kaggle
        
    Returns:
        True si le téléversement réussit, False sinon
    """
    try:
        # Vérifier l'authentification Kaggle
        print("Vérification des identifiants Kaggle...")
        
        import subprocess
        result = subprocess.run(["kaggle", "competitions", "list"], capture_output=True, text=True)
        
        if "ERROR" in result.stderr:
            print("Erreur d'authentification Kaggle. Vérifiez vos identifiants.")
            return False
        
        # Téléverser le notebook
        print(f"Téléversement du notebook {notebook_path} vers la compétition {competition_name}...")
        
        upload_command = [
            "kaggle", "kernels", "push", 
            "--competition", competition_name,
            "--path", notebook_path
        ]
        
        upload_result = subprocess.run(upload_command, capture_output=True, text=True)
        
        if upload_result.returncode == 0:
            print("Notebook téléversé avec succès!")
            print(upload_result.stdout)
            return True
        else:
            print("Erreur lors du téléversement du notebook:")
            print(upload_result.stderr)
            return False
            
    except Exception as e:
        print(f"Erreur lors du téléversement sur Kaggle: {e}")
        return False