"""
Optimisations pour le système Neurax3 ARC Prize 2025
Ce fichier contient les optimisations nécessaires pour traiter la totalité
des puzzles ARC sans limitations et avec sauvegarde automatique.
"""

import os
import time
import json
import torch
import numpy as np
from tqdm import tqdm

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
                    grid_size=64,  # Augmenter la taille de grille
                    time_steps=16,  # Augmenter les pas de temps
                    batch_size=8,   # Traitement par lots
                    precision="float16"  # Précision mixte pour GPU
                )
                print("Moteur Neurax configuré pour utilisation optimale du GPU")
            elif hasattr(engine, "use_gpu"):
                engine.use_gpu = True
                engine.grid_size = 64
                engine.time_steps = 16
                print("Moteur Neurax configuré pour GPU avec paramètres de base")
            return True
    except (ImportError, AttributeError, RuntimeError) as e:
        print(f"Erreur lors de la configuration GPU: {e}")
    
    print("Utilisation du CPU uniquement")
    return False

def process_puzzles_optimized(puzzles, engine, max_time_per_puzzle=600, phase="test", verify_solutions=False):
    """
    Version optimisée de process_puzzles avec:
    - Sauvegarde des points de reprise
    - Support GPU
    - Pas de limite d'époques
    - Sauvegarde des résultats intermédiaires
    """
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
                print(f"Chargé {len(results)} résultats existants")
        except json.JSONDecodeError:
            print(f"Fichier {results_file} corrompu, démarrage d'une nouvelle analyse")
    
    # Filtrer les puzzles déjà traités
    puzzles_to_process = [p for p in puzzles if p.get("id", "unknown") not in processed_ids]
    print(f"Puzzles à traiter: {len(puzzles_to_process)}/{len(puzzles)} ({phase})")
    
    # Statistiques globales
    total_puzzles = len(puzzles)
    total_processed = len(processed_ids)
    total_time = 0
    total_iterations = 0
    successful_puzzles = 0
    
    for puzzle in tqdm(puzzles_to_process, desc=f"Traitement {phase}"):
        start_time = time.time()
        puzzle_id = puzzle.get("id", "unknown")
        
        # Afficher la progression
        total_processed += 1
        progress = (total_processed / total_puzzles) * 100
        remaining = total_puzzles - total_processed
        print(f"\nPuzzle {puzzle_id} ({total_processed}/{total_puzzles}, {progress:.2f}%) - Restants: {remaining}")
        
        try:
            # Convertir au format Neurax
            neurax_puzzle = convert_to_neurax_format(puzzle)
            
            if neurax_puzzle is None:
                raise ValueError(f"Échec de conversion pour le puzzle {puzzle_id}")
            
            # Traiter avec le moteur sans limite d'époques
            result = engine.process_puzzle(
                neurax_puzzle,
                max_time=max_time_per_puzzle,
                max_epochs=0,  # Pas de limite d'époques
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

def optimiser_traitement_neurax3():
    """
    Fonction principale pour remplacer le traitement actuel par la version optimisée
    qui traite tous les puzzles sans limitation.
    """
    print("Optimisation du traitement Neurax3 pour ARC Prize 2025")
    print("========================================================")
    print("1. Traitement des puzzles d'entraînement (1000)")
    print("2. Traitement des puzzles d'évaluation (120)")
    print("3. Traitement des puzzles de test (240)")
    print("========================================================")
    
    # Ces fonctions doivent être appelées depuis le notebook
    print("Pour intégrer ces optimisations dans le notebook, utilisez:")
    print("from neurax3_optimisations import process_puzzles_optimized")
    print("")
    print("# Traitement des puzzles d'entraînement")
    print("training_puzzles = load_training_puzzles(max_puzzles=1000)")
    print("training_results, training_summary = process_puzzles_optimized(")
    print("    training_puzzles, engine, max_time_per_puzzle=600, phase=\"training\", verify_solutions=True)")
    print("")
    print("# Traitement des puzzles d'évaluation")
    print("evaluation_puzzles = load_evaluation_puzzles(max_puzzles=120)")
    print("evaluation_results, evaluation_summary = process_puzzles_optimized(")
    print("    evaluation_puzzles, engine, max_time_per_puzzle=600, phase=\"evaluation\", verify_solutions=True)")
    print("")
    print("# Traitement des puzzles de test")
    print("test_puzzles = load_all_puzzles()")
    print("test_results, test_summary = process_puzzles_optimized(")
    print("    test_puzzles, engine, max_time_per_puzzle=600, phase=\"test\")")

if __name__ == "__main__":
    optimiser_traitement_neurax3()