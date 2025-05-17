#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de validation à grande échelle pour Neurax2
Exécute des tests sur un grand nombre de puzzles ARC pour confirmer les performances
"""

import os
import sys
import json
import time
import logging
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"neurax_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("NeuraxValidation")

# Imports spécifiques à Neurax
try:
    from neurax_engine import NeuraxEngine
    logger.info("Moteur Neurax importé avec succès")
except ImportError:
    logger.error("Erreur lors de l'importation du moteur Neurax")
    raise

def run_validation(training_puzzles=50, evaluation_puzzles=10, test_puzzles=10, batch_size=10):
    """
    Exécute une validation à grande échelle sur un nombre important de puzzles
    
    Args:
        training_puzzles: Nombre de puzzles d'entraînement à traiter
        evaluation_puzzles: Nombre de puzzles d'évaluation à traiter
        test_puzzles: Nombre de puzzles de test à traiter
        batch_size: Taille des lots pour le traitement
    """
    # Créer les répertoires de sortie
    output_dir = f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Créer le moteur Neurax
    engine = NeuraxEngine(
        arc_data_path="./neurax_complet/arc_data",
        default_grid_size=32,
        time_steps=8,
        use_gpu=False,  # Désactivé pour compatibilité
        use_cache=True
    )
    
    # Statistiques globales
    stats = {
        "start_time": time.time(),
        "phases": {},
        "total_puzzles": 0,
        "total_success": 0,
        "overall_success_rate": 0.0,
        "total_duration": 0.0
    }
    
    # Traitement de la phase d'entraînement
    if training_puzzles > 0:
        process_phase(engine, "training", training_puzzles, batch_size, output_dir, stats)
    
    # Traitement de la phase d'évaluation
    if evaluation_puzzles > 0:
        process_phase(engine, "evaluation", evaluation_puzzles, batch_size, output_dir, stats)
    
    # Traitement de la phase de test
    if test_puzzles > 0:
        process_phase(engine, "test", test_puzzles, batch_size, output_dir, stats)
    
    # Finaliser les statistiques globales
    stats["end_time"] = time.time()
    stats["total_duration"] = stats["end_time"] - stats["start_time"]
    
    if stats["total_puzzles"] > 0:
        stats["overall_success_rate"] = (stats["total_success"] / stats["total_puzzles"]) * 100
    
    # Enregistrer les statistiques globales
    with open(os.path.join(output_dir, "global_stats.json"), 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Générer un rapport
    generate_report(stats, output_dir)
    
    logger.info(f"Validation terminée: {stats['total_success']}/{stats['total_puzzles']} puzzles réussis ({stats['overall_success_rate']:.1f}%)")
    logger.info(f"Durée totale: {stats['total_duration']:.2f}s")
    logger.info(f"Résultats enregistrés dans: {output_dir}")

def process_phase(engine, phase, max_puzzles, batch_size, output_dir, stats):
    """
    Traite une phase spécifique de puzzles
    
    Args:
        engine: Moteur Neurax
        phase: Phase à traiter (training/evaluation/test)
        max_puzzles: Nombre maximum de puzzles à traiter
        batch_size: Taille des lots
        output_dir: Répertoire de sortie
        stats: Dictionnaire de statistiques à mettre à jour
    """
    logger.info(f"=== Traitement de la phase {phase} ({max_puzzles} puzzles) ===")
    
    # Statistiques de la phase
    phase_stats = {
        "total": 0,
        "success": 0,
        "success_rate": 0.0,
        "start_time": time.time(),
        "end_time": None,
        "duration": 0.0
    }
    
    # Obtenir les identifiants des puzzles
    puzzle_ids = engine.get_puzzle_ids(phase)
    
    if max_puzzles < len(puzzle_ids):
        puzzle_ids = puzzle_ids[:max_puzzles]
    
    total = len(puzzle_ids)
    phase_stats["total"] = total
    
    # Traiter les puzzles par lots
    all_results = []
    for i in range(0, total, batch_size):
        batch = puzzle_ids[i:i+batch_size]
        batch_size_actual = len(batch)
        
        logger.info(f"Traitement du lot {(i//batch_size)+1}/{(total+batch_size-1)//batch_size} ({batch_size_actual} puzzles)")
        
        # Traiter le lot
        try:
            start_time = time.time()
            batch_results = []
            
            # Traiter chaque puzzle individuellement pour éviter les erreurs de processus
            for puzzle_id in batch:
                try:
                    result = engine.process_single_puzzle(puzzle_id, phase)
                    batch_results.append(result)
                    
                    # Mettre à jour les statistiques
                    if result.get("status") == "PASS":
                        phase_stats["success"] += 1
                        stats["total_success"] += 1
                        
                    # Afficher la progression individuelle
                    status = result.get("status", "UNKNOWN")
                    duration = result.get("duration", 0)
                    logger.info(f"Puzzle {puzzle_id} traité: {status} ({duration:.4f}s)")
                    
                except Exception as e:
                    logger.error(f"Erreur lors du traitement du puzzle {puzzle_id}: {str(e)}")
                    batch_results.append({
                        "puzzle_id": puzzle_id,
                        "phase": phase,
                        "status": "FAIL",
                        "error": str(e)
                    })
            
            # Ajouter les résultats du lot
            all_results.extend(batch_results)
            
            # Afficher la progression
            processed = i + batch_size_actual
            success_rate = (phase_stats["success"] / processed) * 100 if processed > 0 else 0
            duration = time.time() - start_time
            
            logger.info(f"Progression: {processed}/{total} ({processed/total*100:.1f}%) - Taux de réussite: {success_rate:.1f}% - Durée du lot: {duration:.2f}s")
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement du lot: {str(e)}")
    
    # Finaliser les statistiques de la phase
    phase_stats["end_time"] = time.time()
    phase_stats["duration"] = phase_stats["end_time"] - phase_stats["start_time"]
    
    if phase_stats["total"] > 0:
        phase_stats["success_rate"] = (phase_stats["success"] / phase_stats["total"]) * 100
    
    # Mettre à jour les statistiques globales
    stats["phases"][phase] = phase_stats
    stats["total_puzzles"] += phase_stats["total"]
    
    # Enregistrer les résultats
    results_file = os.path.join(output_dir, f"{phase}_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Enregistrer les statistiques de la phase
    stats_file = os.path.join(output_dir, f"{phase}_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(phase_stats, f, indent=2)
    
    logger.info(f"=== Résumé de la phase {phase} ===")
    logger.info(f"Puzzles traités: {phase_stats['total']}")
    logger.info(f"Puzzles réussis: {phase_stats['success']} ({phase_stats['success_rate']:.1f}%)")
    logger.info(f"Durée totale: {phase_stats['duration']:.2f}s")
    logger.info(f"Durée moyenne par puzzle: {phase_stats['duration']/phase_stats['total']:.4f}s")

def generate_report(stats, output_dir):
    """
    Génère un rapport détaillé des résultats de validation
    
    Args:
        stats: Statistiques globales
        output_dir: Répertoire de sortie
    """
    # Créer le contenu du rapport
    report = f"""# Rapport de Validation à Grande Échelle de Neurax2

## Résumé Exécutif

Ce rapport présente les résultats de la validation à grande échelle du système Neurax2 sur un ensemble significatif de puzzles ARC. Au total, {stats['total_puzzles']} puzzles ont été traités, avec un taux de réussite global de {stats['overall_success_rate']:.1f}%.

## Statistiques Globales

- **Puzzles traités**: {stats['total_puzzles']}
- **Puzzles réussis**: {stats['total_success']} ({stats['overall_success_rate']:.1f}%)
- **Durée totale**: {stats['total_duration']:.2f} secondes
- **Durée moyenne par puzzle**: {stats['total_duration']/stats['total_puzzles']:.4f} secondes (si total > 0)

## Résultats par Phase

"""
    
    # Ajouter les résultats pour chaque phase
    for phase, phase_stats in stats["phases"].items():
        report += f"""### Phase {phase.capitalize()}

- **Puzzles traités**: {phase_stats['total']}
- **Puzzles réussis**: {phase_stats['success']} ({phase_stats['success_rate']:.1f}%)
- **Durée totale**: {phase_stats['duration']:.2f} secondes
- **Durée moyenne par puzzle**: {phase_stats['duration']/phase_stats['total']:.4f} secondes (si total > 0)

"""
    
    # Ajouter la conclusion
    report += f"""## Conclusion

La validation à grande échelle de Neurax2 montre {"d'excellents" if stats['overall_success_rate'] > 95 else "de bons" if stats['overall_success_rate'] > 80 else "des" } résultats avec un taux de réussite global de {stats['overall_success_rate']:.1f}%. {"Le système est prêt pour le traitement de l'ensemble des 1360 puzzles de la compétition ARC-Prize-2025." if stats['overall_success_rate'] > 90 else "Des améliorations sont encore nécessaires avant de traiter l'ensemble des puzzles de la compétition."}

---

*Rapport généré le {datetime.now().strftime('%Y-%m-%d à %H:%M:%S')}*
"""
    
    # Enregistrer le rapport
    report_file = os.path.join(output_dir, "validation_report.md")
    with open(report_file, 'w') as f:
        f.write(report)
    
    logger.info(f"Rapport de validation généré: {report_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validation à grande échelle de Neurax2")
    parser.add_argument("--training", type=int, default=50, help="Nombre de puzzles d'entraînement")
    parser.add_argument("--evaluation", type=int, default=10, help="Nombre de puzzles d'évaluation")
    parser.add_argument("--test", type=int, default=10, help="Nombre de puzzles de test")
    parser.add_argument("--batch-size", type=int, default=10, help="Taille des lots")
    
    args = parser.parse_args()
    
    run_validation(
        training_puzzles=args.training,
        evaluation_puzzles=args.evaluation,
        test_puzzles=args.test,
        batch_size=args.batch_size
    )