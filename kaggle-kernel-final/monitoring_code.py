"""
Code de surveillance pour Neurax3 - À ajouter au notebook Kaggle
"""

# Cellule de surveillance à ajouter au notebook
import os
import time
import threading
import logging
import datetime
import json
import sys
import glob

# Créer les répertoires nécessaires
os.makedirs("logs", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Configuration du logger
log_filename = f"neurax3_execution_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_path = os.path.join("logs", log_filename)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('neurax3_monitor')

logger.info("=== DÉMARRAGE DE L'OPTIMISATION NEURAX3 POUR ARC-PRIZE-2025 ===")
logger.info(f"Version: 1.2 - Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("Mode: Traitement complet sans limitation (1360 puzzles)")

# Fonction pour surveiller l'avancement et générer des rapports
def monitor_progress():
    """Fonction de surveillance qui s'exécute en arrière-plan"""
    while True:
        try:
            # Rechercher tous les fichiers de checkpoint
            checkpoint_files = glob.glob("*_checkpoint.json")
            if checkpoint_files:
                progress_data = {
                    "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "phases": {}
                }
                
                # Pour chaque phase (training, evaluation, test)
                for checkpoint_file in checkpoint_files:
                    phase = checkpoint_file.split("_")[0]
                    try:
                        with open(checkpoint_file, 'r') as f:
                            checkpoint = json.load(f)
                        
                        # Récupérer les IDs des puzzles traités
                        processed_ids = checkpoint.get("processed_ids", [])
                        
                        # Définir le nombre total de puzzles par phase
                        total_puzzles = 0
                        if phase == "training":
                            total_puzzles = 1000
                        elif phase == "evaluation":
                            total_puzzles = 120
                        elif phase == "test":
                            total_puzzles = 240
                        
                        # Calculer la progression
                        progress = len(processed_ids) / total_puzzles * 100 if total_puzzles > 0 else 0
                        
                        # Ajouter aux données de progression
                        progress_data["phases"][phase] = {
                            "processed": len(processed_ids),
                            "total": total_puzzles,
                            "progress": progress
                        }
                        
                        logger.info(f"Phase {phase}: {len(processed_ids)}/{total_puzzles} puzzles ({progress:.2f}%)")
                    except Exception as e:
                        logger.error(f"Erreur lors de la lecture du checkpoint {checkpoint_file}: {e}")
                
                # Calculer la progression globale
                total_processed = sum(progress_data["phases"].get(phase, {}).get("processed", 0) for phase in progress_data["phases"])
                total_puzzles = sum(progress_data["phases"].get(phase, {}).get("total", 0) for phase in progress_data["phases"])
                
                if total_puzzles > 0:
                    global_progress = total_processed / total_puzzles * 100
                else:
                    global_progress = 0
                
                progress_data["global"] = {
                    "processed": total_processed,
                    "total": total_puzzles,
                    "progress": global_progress
                }
                
                # Enregistrer le rapport de progression
                report_path = os.path.join("reports", f"progress_{int(time.time())}.json")
                with open(report_path, 'w') as f:
                    json.dump(progress_data, f, indent=2)
                
                # Générer un rapport en texte
                report_text_path = os.path.join("reports", f"progress_{int(time.time())}.txt")
                with open(report_text_path, 'w') as f:
                    f.write("=== RAPPORT DE PROGRESSION NEURAX3 ===\n\n")
                    f.write(f"Date: {progress_data['timestamp']}\n\n")
                    
                    f.write(f"PROGRESSION GLOBALE: {global_progress:.2f}%\n")
                    f.write(f"Puzzles traités: {total_processed}/{total_puzzles}\n\n")
                    
                    for phase in progress_data["phases"]:
                        phase_data = progress_data["phases"][phase]
                        f.write(f"Phase {phase}:\n")
                        f.write(f"  Puzzles traités: {phase_data['processed']}/{phase_data['total']} ({phase_data['progress']:.2f}%)\n")
                        
                        # Ajouter les informations de réussite si disponibles
                        summary_file = f"{phase}_summary.json"
                        if os.path.exists(summary_file):
                            try:
                                with open(summary_file, 'r') as summary_f:
                                    summary = json.load(summary_f)
                                    success_rate = summary.get("success_rate", 0)
                                    f.write(f"  Taux de réussite: {success_rate:.2f}%\n")
                            except Exception as e:
                                logger.error(f"Erreur lors de la lecture du résumé {summary_file}: {e}")
                        
                        f.write("\n")
                
                logger.info(f"Rapport de progression généré: {report_text_path}")
                
                # Mettre à jour le statut global
                status_path = os.path.join("logs", "status.txt")
                with open(status_path, 'w') as f:
                    f.write("=== STATUT D'EXÉCUTION NEURAX3 ===\n\n")
                    f.write(f"Dernière mise à jour: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Progression globale: {global_progress:.2f}%\n")
                    f.write(f"Puzzles traités: {total_processed}/{total_puzzles}\n\n")
                    
                    for phase in progress_data["phases"]:
                        phase_data = progress_data["phases"][phase]
                        f.write(f"Phase {phase}: {phase_data['processed']}/{phase_data['total']} ({phase_data['progress']:.2f}%)\n")
                    
                    f.write("\nCe fichier est mis à jour automatiquement toutes les 5 minutes.")
            
            # Attendre avant la prochaine vérification
            time.sleep(300)  # 5 minutes
        except Exception as e:
            logger.error(f"Erreur dans la surveillance: {e}")
            time.sleep(60)  # Attendre 1 minute en cas d'erreur

# Démarrer le thread de surveillance
logger.info("Démarrage du thread de surveillance...")
progress_thread = threading.Thread(target=monitor_progress)
progress_thread.daemon = True
progress_thread.start()

print("=== SYSTÈME DE SURVEILLANCE NEURAX3 ACTIVÉ ===")
print("Le système surveille l'avancement du traitement des puzzles.")
print("Les logs et rapports sont générés automatiquement.")
print(f"Fichier de log principal: {log_path}")
print("Dossier des rapports: reports/")
print("Le statut actuel est disponible dans: logs/status.txt")