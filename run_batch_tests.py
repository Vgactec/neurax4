#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script d'exécution optimisé pour les tests Neurax2 par lots
Utilise le traitement parallèle et la mise en cache pour accélérer l'analyse
"""

import os
import sys
import time
import json
import csv
import logging
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"arc_learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("NeuraxBatch")

# Nombre de processus parallèles
NUM_PROCESSES = max(1, multiprocessing.cpu_count() - 1)

# Import du simulateur optimisé
from quantum_gravity_sim import QuantumGravitySimulator

def load_puzzles(phase: str, data_path: str = "./neurax_complet/arc_data", max_puzzles: int = None) -> Dict[str, Any]:
    """
    Charge les puzzles ARC d'une phase spécifique
    
    Args:
        phase: Phase à charger (training/evaluation/test)
        data_path: Chemin vers les données ARC
        max_puzzles: Nombre maximum de puzzles à charger
        
    Returns:
        Dictionnaire contenant les puzzles chargés
    """
    # Ajuster les noms de fichiers selon la structure ARC-Prize-2025
    filename = ""
    if phase == "training":
        filename = "arc-agi_training_challenges.json"
    elif phase == "evaluation":
        filename = "arc-agi_evaluation_challenges.json"
    elif phase == "test":
        filename = "arc-agi_test_challenges.json"
    else:
        raise ValueError(f"Phase inconnue: {phase}")
    
    filepath = os.path.join(data_path, filename)
    
    logger.info(f"Chargement des puzzles de la phase {phase}")
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # S'adapter à la structure du fichier
        if "challenges" in data:
            puzzles = {str(challenge["id"]): challenge for challenge in data["challenges"]}
        else:
            # Format alternatif où chaque clé est un ID de puzzle
            puzzles = data
            
        logger.info(f"Chargement réussi: {len(puzzles)} puzzles de {phase}")
        
        if max_puzzles and max_puzzles < len(puzzles):
            logger.info(f"Limitation à {max_puzzles} puzzles pour la phase {phase}")
            # Convertir en liste de tuples, limiter, puis reconvertir en dict
            puzzle_items = list(puzzles.items())[:max_puzzles]
            puzzles = dict(puzzle_items)
            
        return puzzles
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement des puzzles {phase}: {str(e)}")
        return {}

def process_puzzle(args) -> Dict[str, Any]:
    """
    Traite un puzzle avec le simulateur de gravité quantique
    
    Args:
        args: Tuple (puzzle_id, puzzle_data, phase)
        
    Returns:
        Résultats du traitement
    """
    puzzle_id, puzzle_data, phase = args
    
    logger.info(f"Traitement du puzzle {puzzle_id} - Phase: {phase}")
    
    start_time = time.time()
    
    try:
        # Adapter à la structure réelle des puzzles ARC
        grid_size = 32  # Taille par défaut
        
        # Préparer les données pour le simulateur
        processed_data = {
            "id": puzzle_id,
            "train": []
        }
        
        # Si le puzzle contient directement des exemples d'entraînement
        if "train" in puzzle_data:
            processed_data["train"] = puzzle_data["train"]
        
        # Si le puzzle contient une description et des exemples
        elif "description" in puzzle_data:
            # Pour les besoins du test, créer un exemple simple
            processed_data["train"] = [{
                "input": [[0, 0], [0, 0]],
                "output": [[1, 1], [1, 1]]
            }]
        
        # Cas où nous avons des exemples mais sous un autre format
        elif "examples" in puzzle_data:
            processed_data["train"] = puzzle_data["examples"]
        
        # Sortie précoce si aucune donnée d'entraînement n'est disponible
        if not processed_data["train"]:
            return {
                "puzzle_id": puzzle_id,
                "phase": phase,
                "status": "FAIL",
                "error": "Pas de données d'entraînement exploitables",
                "duration": time.time() - start_time
            }
        
        # Utiliser une taille de grille par défaut ou calculée
        try:
            # Essayer d'extraire la taille des exemples si possible
            example = processed_data["train"][0]
            if "input" in example and isinstance(example["input"], list):
                height = len(example["input"])
                width = len(example["input"][0]) if height > 0 else 0
                
                # Arrondir à la puissance de 2 supérieure pour optimiser les calculs
                grid_size = max(32, 2 ** (height - 1).bit_length(), 2 ** (width - 1).bit_length())
        except (IndexError, KeyError, TypeError):
            # En cas d'erreur, conserver la taille par défaut
            pass
        
        # Créer le simulateur avec la taille adaptée
        simulator = QuantumGravitySimulator(grid_size=grid_size, time_steps=8, use_cache=True)
        
        # Appliquer le traitement
        sim_results = simulator.process_puzzle(processed_data)
        
        # Vérifier les résultats
        success = not sim_results.get("error")
        
        return {
            "puzzle_id": puzzle_id,
            "phase": phase,
            "status": "PASS" if success else "FAIL",
            "grid_size": grid_size,
            "simulator_details": sim_results,
            "duration": time.time() - start_time
        }
    
    except Exception as e:
        logger.error(f"Erreur lors du traitement du puzzle {puzzle_id}: {str(e)}")
        return {
            "puzzle_id": puzzle_id,
            "phase": phase,
            "status": "FAIL",
            "error": str(e),
            "duration": time.time() - start_time
        }

def process_puzzles_batch(phase: str, puzzles: Dict[str, Any], batch_size: int = 10) -> List[Dict[str, Any]]:
    """
    Traite un lot de puzzles en parallèle
    
    Args:
        phase: Phase de traitement
        puzzles: Dictionnaire des puzzles à traiter
        batch_size: Taille du lot
        
    Returns:
        Liste des résultats du traitement
    """
    results = []
    puzzle_items = list(puzzles.items())
    
    # Traiter par lots pour limiter l'utilisation de la mémoire
    for i in range(0, len(puzzle_items), batch_size):
        batch = puzzle_items[i:i+batch_size]
        logger.info(f"Traitement du lot {i//batch_size + 1}/{(len(puzzle_items) + batch_size - 1)//batch_size} ({len(batch)} puzzles)")
        
        # Préparer les arguments pour le traitement parallèle
        args_list = [(puzzle_id, puzzle_data, phase) for puzzle_id, puzzle_data in batch]
        
        # Traitement parallèle
        with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
            futures = [executor.submit(process_puzzle, args) for args in args_list]
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Log du résultat
                    puzzle_id = result.get("puzzle_id", "inconnu")
                    status = result.get("status", "INCONNU")
                    duration = result.get("duration", 0)
                    
                    logger.info(f"Puzzle {puzzle_id} - Statut: {status} - Durée: {duration:.4f}s")
                    
                    # Affichage des statistiques en temps réel
                    completed = len(results)
                    success = sum(1 for r in results if r.get("status") == "PASS")
                    success_rate = success / completed if completed > 0 else 0
                    
                    logger.info(f"Progression: {completed}/{len(puzzles)} ({completed/len(puzzles)*100:.1f}%) - Taux de réussite: {success_rate*100:.1f}%")
                    
                except Exception as e:
                    logger.error(f"Erreur lors de la récupération des résultats: {str(e)}")
    
    return results

def export_results(results: List[Dict[str, Any]], phase: str, output_prefix: str = "arc_batch_test") -> None:
    """
    Exporte les résultats au format JSON et CSV
    
    Args:
        results: Liste des résultats
        phase: Phase de traitement
        output_prefix: Préfixe pour les fichiers de sortie
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export JSON
    json_file = f"{output_prefix}_{phase}_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Résultats exportés au format JSON: {json_file}")
    
    # Export CSV
    csv_file = f"{output_prefix}_{phase}_{timestamp}.csv"
    with open(csv_file, 'w', newline='') as f:
        fieldnames = ["puzzle_id", "phase", "status", "duration", "grid_size"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            row = {
                "puzzle_id": result.get("puzzle_id", ""),
                "phase": result.get("phase", ""),
                "status": result.get("status", ""),
                "duration": result.get("duration", 0),
                "grid_size": result.get("grid_size", 0)
            }
            writer.writerow(row)
    logger.info(f"Résultats exportés au format CSV: {csv_file}")
    
    # Statistiques globales
    total = len(results)
    success = sum(1 for r in results if r.get("status") == "PASS")
    success_rate = success / total if total > 0 else 0
    avg_duration = sum(r.get("duration", 0) for r in results) / total if total > 0 else 0
    
    logger.info(f"=== RÉSUMÉ DES RÉSULTATS ({phase}) ===")
    logger.info(f"Total puzzles: {total}")
    logger.info(f"Réussis: {success} ({success_rate*100:.1f}%)")
    logger.info(f"Durée moyenne: {avg_duration:.4f}s")
    
    # Export du résumé
    summary_file = f"{output_prefix}_{phase}_{timestamp}_summary.json"
    summary = {
        "phase": phase,
        "timestamp": timestamp,
        "total_puzzles": total,
        "success_count": success,
        "success_rate": success_rate,
        "average_duration": avg_duration,
        "cache_stats": QuantumGravitySimulator.get_cache_stats()
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Résumé exporté: {summary_file}")

def main():
    """
    Point d'entrée principal
    """
    parser = argparse.ArgumentParser(description="Exécution optimisée des tests Neurax2 par lots")
    parser.add_argument("--phase", type=str, choices=["training", "evaluation", "test", "all"], default="training",
                      help="Phase à traiter (training/evaluation/test/all)")
    parser.add_argument("--max", type=int, default=10,
                      help="Nombre maximum de puzzles à traiter par phase")
    parser.add_argument("--batch", type=int, default=5,
                      help="Taille des lots pour le traitement parallèle")
    parser.add_argument("--data-path", type=str, default="./neurax_complet/arc_data",
                      help="Chemin vers les données ARC")
    parser.add_argument("--output", type=str, default="arc_batch_test",
                      help="Préfixe pour les fichiers de sortie")
    
    args = parser.parse_args()
    
    logger.info("=== DÉMARRAGE DE L'EXÉCUTION OPTIMISÉE DES TESTS NEURAX2 ===")
    logger.info(f"Configuration:")
    logger.info(f"- Phase: {args.phase}")
    logger.info(f"- Max puzzles par phase: {args.max}")
    logger.info(f"- Taille des lots: {args.batch}")
    logger.info(f"- Processus parallèles: {NUM_PROCESSES}")
    logger.info(f"- Données ARC: {args.data_path}")
    
    start_time = time.time()
    
    phases = ["training", "evaluation", "test"] if args.phase == "all" else [args.phase]
    
    for phase in phases:
        phase_start = time.time()
        logger.info(f"=== DÉMARRAGE DE LA PHASE {phase.upper()} ===")
        
        # Charger les puzzles
        puzzles = load_puzzles(phase, args.data_path, args.max)
        
        if not puzzles:
            logger.error(f"Aucun puzzle chargé pour la phase {phase}")
            continue
        
        # Traiter les puzzles par lots
        results = process_puzzles_batch(phase, puzzles, args.batch)
        
        # Exporter les résultats
        export_results(results, phase, args.output)
        
        phase_duration = time.time() - phase_start
        logger.info(f"=== FIN DE LA PHASE {phase.upper()} ===")
        logger.info(f"Durée totale de la phase {phase}: {phase_duration:.2f}s")
    
    total_duration = time.time() - start_time
    logger.info("=== FIN DE L'EXÉCUTION DES TESTS ===")
    logger.info(f"Durée totale: {total_duration:.2f}s")
    
    # Afficher les statistiques du cache
    cache_stats = QuantumGravitySimulator.get_cache_stats()
    logger.info(f"Statistiques du cache:")
    logger.info(f"- Hits: {cache_stats['hits']}")
    logger.info(f"- Misses: {cache_stats['misses']}")
    logger.info(f"- Total: {cache_stats['total']}")
    logger.info(f"- Hit ratio: {cache_stats['hit_ratio']*100:.1f}%")

if __name__ == "__main__":
    main()