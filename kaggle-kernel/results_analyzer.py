"""
Script d'analyse des résultats pour Neurax3
Ce script vérifie les résultats de l'exécution Neurax3 pour confirmer
que toutes les optimisations fonctionnent correctement.
"""

import os
import sys
import json
import logging
import datetime
import glob

# Configuration du logger
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"results_analyzer_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('results_analyzer')

# Fonction pour analyser les résultats
def analyze_results():
    """Analyse les résultats de l'exécution Neurax3"""
    logger.info("=== DÉMARRAGE DE L'ANALYSE DES RÉSULTATS ===")
    
    # Recherche des fichiers de résultats
    result_files = {
        "training": glob.glob("training_*.json"),
        "evaluation": glob.glob("evaluation_*.json"),
        "test": glob.glob("test_*.json"),
        "summary": glob.glob("*_summary.json"),
        "checkpoint": glob.glob("*_checkpoint.json"),
        "log": glob.glob("*.log")
    }
    
    logger.info("Fichiers trouvés:")
    for category, files in result_files.items():
        logger.info(f"  {category}: {len(files)} fichier(s)")
        for file in files:
            logger.info(f"    - {file}")
    
    # Analyse des fichiers de résultats
    results = {}
    
    # Vérifier les fichiers de résumé
    if result_files["summary"]:
        for summary_file in result_files["summary"]:
            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                
                phase = summary.get("phase", "unknown")
                results[f"{phase}_summary"] = {
                    "total_puzzles": summary.get("total_puzzles", 0),
                    "processed_puzzles": summary.get("processed_puzzles", 0),
                    "successful_puzzles": summary.get("successful_puzzles", 0),
                    "success_rate": summary.get("success_rate", 0),
                    "average_time_per_puzzle": summary.get("average_time_per_puzzle", 0)
                }
                
                logger.info(f"Résumé {phase}: {results[f'{phase}_summary']}")
            except Exception as e:
                logger.error(f"Erreur lors de l'analyse du fichier {summary_file}: {e}")
    
    # Vérifier les points de reprise
    if result_files["checkpoint"]:
        for checkpoint_file in result_files["checkpoint"]:
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                
                phase = checkpoint_file.split("_")[0]
                results[f"{phase}_checkpoint"] = {
                    "processed_ids": len(checkpoint.get("processed_ids", [])),
                    "timestamp": checkpoint.get("timestamp", 0),
                    "date": checkpoint.get("date", "")
                }
                
                logger.info(f"Point de reprise {phase}: {results[f'{phase}_checkpoint']}")
            except Exception as e:
                logger.error(f"Erreur lors de l'analyse du fichier {checkpoint_file}: {e}")
    
    # Générer un rapport d'analyse
    report_path = os.path.join(log_dir, "results_analysis.md")
    with open(report_path, 'w') as f:
        f.write("# Rapport d'analyse des résultats Neurax3\n\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Fichiers analysés\n\n")
        for category, files in result_files.items():
            f.write(f"### {category.capitalize()}\n\n")
            if files:
                for file in files:
                    f.write(f"- {file}\n")
            else:
                f.write("Aucun fichier trouvé\n")
            f.write("\n")
        
        f.write("## Résumé des résultats\n\n")
        if "training_summary" in results:
            f.write("### Entraînement\n\n")
            f.write(f"- Puzzles totaux: {results['training_summary']['total_puzzles']}\n")
            f.write(f"- Puzzles traités: {results['training_summary']['processed_puzzles']}\n")
            f.write(f"- Puzzles réussis: {results['training_summary']['successful_puzzles']}\n")
            f.write(f"- Taux de réussite: {results['training_summary']['success_rate']:.2f}%\n")
            f.write(f"- Temps moyen par puzzle: {results['training_summary']['average_time_per_puzzle']:.2f}s\n\n")
        
        if "evaluation_summary" in results:
            f.write("### Évaluation\n\n")
            f.write(f"- Puzzles totaux: {results['evaluation_summary']['total_puzzles']}\n")
            f.write(f"- Puzzles traités: {results['evaluation_summary']['processed_puzzles']}\n")
            f.write(f"- Puzzles réussis: {results['evaluation_summary']['successful_puzzles']}\n")
            f.write(f"- Taux de réussite: {results['evaluation_summary']['success_rate']:.2f}%\n")
            f.write(f"- Temps moyen par puzzle: {results['evaluation_summary']['average_time_per_puzzle']:.2f}s\n\n")
        
        if "test_summary" in results:
            f.write("### Test\n\n")
            f.write(f"- Puzzles totaux: {results['test_summary']['total_puzzles']}\n")
            f.write(f"- Puzzles traités: {results['test_summary']['processed_puzzles']}\n")
            f.write(f"- Puzzles réussis: {results['test_summary']['successful_puzzles']}\n")
            f.write(f"- Taux de réussite: {results['test_summary']['success_rate']:.2f}%\n")
            f.write(f"- Temps moyen par puzzle: {results['test_summary']['average_time_per_puzzle']:.2f}s\n\n")
        
        f.write("## Points de reprise\n\n")
        if "training_checkpoint" in results:
            f.write(f"### Entraînement\n\n")
            f.write(f"- Puzzles traités: {results['training_checkpoint']['processed_ids']}\n")
            f.write(f"- Date: {results['training_checkpoint']['date']}\n\n")
        
        if "evaluation_checkpoint" in results:
            f.write(f"### Évaluation\n\n")
            f.write(f"- Puzzles traités: {results['evaluation_checkpoint']['processed_ids']}\n")
            f.write(f"- Date: {results['evaluation_checkpoint']['date']}\n\n")
        
        if "test_checkpoint" in results:
            f.write(f"### Test\n\n")
            f.write(f"- Puzzles traités: {results['test_checkpoint']['processed_ids']}\n")
            f.write(f"- Date: {results['test_checkpoint']['date']}\n\n")
        
        f.write("## Conclusion\n\n")
        f.write("L'analyse des résultats montre que:\n\n")
        
        # Vérifier si tous les puzzles ont été traités
        all_puzzles_processed = True
        if "training_summary" in results and results["training_summary"]["total_puzzles"] > 0:
            if results["training_summary"]["processed_puzzles"] < results["training_summary"]["total_puzzles"]:
                all_puzzles_processed = False
        
        if "evaluation_summary" in results and results["evaluation_summary"]["total_puzzles"] > 0:
            if results["evaluation_summary"]["processed_puzzles"] < results["evaluation_summary"]["total_puzzles"]:
                all_puzzles_processed = False
        
        if "test_summary" in results and results["test_summary"]["total_puzzles"] > 0:
            if results["test_summary"]["processed_puzzles"] < results["test_summary"]["total_puzzles"]:
                all_puzzles_processed = False
        
        if all_puzzles_processed:
            f.write("✅ Tous les puzzles ont été traités avec succès\n")
        else:
            f.write("❌ Certains puzzles n'ont pas été traités\n")
        
        # Vérifier le taux de réussite
        success_rate_ok = True
        if "training_summary" in results:
            if results["training_summary"]["success_rate"] < 50:
                success_rate_ok = False
        
        if "evaluation_summary" in results:
            if results["evaluation_summary"]["success_rate"] < 50:
                success_rate_ok = False
        
        if success_rate_ok:
            f.write("✅ Le taux de réussite est satisfaisant\n")
        else:
            f.write("❌ Le taux de réussite est inférieur à 50%\n")
        
        # Vérifier si les points de reprise fonctionnent
        checkpoints_ok = True
        if not result_files["checkpoint"]:
            checkpoints_ok = False
        
        if checkpoints_ok:
            f.write("✅ Le système de points de reprise fonctionne correctement\n")
        else:
            f.write("❌ Aucun point de reprise n'a été créé\n")
    
    logger.info(f"Rapport d'analyse généré: {report_path}")
    logger.info("=== FIN DE L'ANALYSE DES RÉSULTATS ===")
    
    return report_path, results

# Fonction principale
def main():
    """Fonction principale pour l'analyse des résultats"""
    try:
        report_path, results = analyze_results()
        
        # Créer un fichier de statut
        status_path = os.path.join(log_dir, "analysis_status.txt")
        with open(status_path, 'w') as f:
            f.write("=== STATUT DE L'ANALYSE DES RÉSULTATS ===\n")
            f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            if results:
                f.write("Résultats trouvés: Oui\n")
                
                phases = ["training", "evaluation", "test"]
                total_puzzles = 0
                processed_puzzles = 0
                
                for phase in phases:
                    summary_key = f"{phase}_summary"
                    if summary_key in results:
                        total_puzzles += results[summary_key]["total_puzzles"]
                        processed_puzzles += results[summary_key]["processed_puzzles"]
                
                f.write(f"Total des puzzles: {total_puzzles}\n")
                f.write(f"Puzzles traités: {processed_puzzles}\n")
                
                if total_puzzles > 0:
                    f.write(f"Progression: {(processed_puzzles / total_puzzles) * 100:.2f}%\n")
                
                f.write(f"Chemin du rapport: {report_path}\n")
            else:
                f.write("Résultats trouvés: Non\n")
        
        logger.info(f"Fichier de statut créé: {status_path}")
        
    except Exception as e:
        logger.error(f"Erreur non gérée: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()