"""
Code de surveillance pour Neurax3 
Ce script permet de surveiller l'exécution du notebook et de générer des rapports.
"""

import os
import time
import threading
import logging
import datetime
import json
import sys
import glob
import shutil

# Créer les répertoires nécessaires
os.makedirs("logs", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Configuration du logger principal
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

# Créer un fichier de statut initial
status_file = os.path.join("logs", "status.txt")
with open(status_file, "w") as f:
    f.write(f"=== STATUT D'EXÉCUTION NEURAX3 ===\n")
    f.write(f"Démarrage: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Mode: TRAITEMENT COMPLET SANS LIMITATION\n")
    f.write(f"Total puzzles: 1360 (1000 training + 120 evaluation + 240 test)\n")
    f.write(f"Fichier de logs: {log_path}\n")

logger.info("=== DÉMARRAGE DE L'ANALYSE NEURAX3 - VERSION OPTIMISÉE ===")
logger.info("Mode: TRAITEMENT COMPLET SANS LIMITATION (1360 puzzles)")
logger.info(f"Date de démarrage: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Fonction pour surveiller les checkpoints
def monitor_checkpoints():
    while True:
        try:
            # Rechercher tous les checkpoints
            checkpoint_files = glob.glob("*_checkpoint.json")
            
            if checkpoint_files:
                status = {"timestamp": time.time()}
                
                for checkpoint_file in checkpoint_files:
                    phase = checkpoint_file.split("_")[0]
                    
                    try:
                        with open(checkpoint_file, "r") as f:
                            checkpoint = json.load(f)
                            
                        processed_ids = checkpoint.get("processed_ids", [])
                        
                        status[phase] = {
                            "processed": len(processed_ids),
                            "last_updated": checkpoint.get("date", "Unknown")
                        }
                        
                        # Sauvegarder une copie du checkpoint dans le dossier logs
                        shutil.copy(checkpoint_file, os.path.join("logs", f"{phase}_checkpoint_{int(time.time())}.json"))
                        
                        logger.info(f"Checkpoint {phase}: {len(processed_ids)} puzzles traités")
                    except Exception as e:
                        logger.error(f"Erreur lors de la lecture du checkpoint {checkpoint_file}: {e}")
                
                # Sauvegarder le statut
                with open(os.path.join("logs", "checkpoint_status.json"), "w") as f:
                    json.dump(status, f, indent=2)
            
            # Attendre avant la prochaine vérification
            time.sleep(300)  # 5 minutes
        except Exception as e:
            logger.error(f"Erreur dans la surveillance des checkpoints: {e}")
            time.sleep(60)  # En cas d'erreur, attendre 1 minute avant de réessayer

# Fonction pour générer périodiquement des rapports d'avancement
def generate_progress_reports():
    while True:
        try:
            # Collecter les informations sur l'avancement
            report = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "phases": {}
            }
            
            # Vérifier tous les fichiers de résultats
            for phase in ["training", "evaluation", "test"]:
                report["phases"][phase] = {
                    "checkpoint_exists": os.path.exists(f"{phase}_checkpoint.json"),
                    "results_exists": os.path.exists(f"{phase}_results.json"),
                    "summary_exists": os.path.exists(f"{phase}_summary.json"),
                    "processed": 0,
                    "total": 0,
                    "success_rate": 0
                }
                
                # Définir les totaux attendus
                if phase == "training":
                    report["phases"][phase]["total"] = 1000
                elif phase == "evaluation":
                    report["phases"][phase]["total"] = 120
                elif phase == "test":
                    report["phases"][phase]["total"] = 240
                
                # Lire le checkpoint
                if report["phases"][phase]["checkpoint_exists"]:
                    try:
                        with open(f"{phase}_checkpoint.json", "r") as f:
                            checkpoint = json.load(f)
                        
                        processed_ids = checkpoint.get("processed_ids", [])
                        report["phases"][phase]["processed"] = len(processed_ids)
                    except Exception as e:
                        logger.error(f"Erreur lors de la lecture du checkpoint {phase}: {e}")
                
                # Lire le résumé
                if report["phases"][phase]["summary_exists"]:
                    try:
                        with open(f"{phase}_summary.json", "r") as f:
                            summary = json.load(f)
                        
                        report["phases"][phase]["success_rate"] = summary.get("success_rate", 0)
                    except Exception as e:
                        logger.error(f"Erreur lors de la lecture du résumé {phase}: {e}")
            
            # Calculer la progression globale
            total_puzzles = sum(report["phases"][phase]["total"] for phase in ["training", "evaluation", "test"])
            processed_puzzles = sum(report["phases"][phase]["processed"] for phase in ["training", "evaluation", "test"])
            
            if total_puzzles > 0:
                report["global_progress"] = (processed_puzzles / total_puzzles) * 100
            else:
                report["global_progress"] = 0
            
            # Sauvegarder le rapport
            report_filename = f"progress_report_{int(time.time())}.json"
            with open(os.path.join("reports", report_filename), "w") as f:
                json.dump(report, f, indent=2)
            
            # Créer un rapport en texte
            report_text_filename = f"progress_report_{int(time.time())}.txt"
            with open(os.path.join("reports", report_text_filename), "w") as f:
                f.write("=== RAPPORT D'AVANCEMENT NEURAX3 ===\n")
                f.write(f"Date: {report['timestamp']}\n\n")
                
                f.write(f"Progression globale: {report['global_progress']:.2f}%\n")
                f.write(f"Puzzles traités: {processed_puzzles}/{total_puzzles}\n\n")
                
                for phase in ["training", "evaluation", "test"]:
                    f.write(f"Phase {phase}:\n")
                    phase_progress = (report['phases'][phase]['processed'] / report['phases'][phase]['total'] * 100) if report['phases'][phase]['total'] > 0 else 0
                    f.write(f"  Puzzles traités: {report['phases'][phase]['processed']}/{report['phases'][phase]['total']} ({phase_progress:.2f}%)\n")
                    f.write(f"  Taux de réussite: {report['phases'][phase]['success_rate']:.2f}%\n\n")
            
            logger.info(f"Rapport d'avancement généré: {report_text_filename}")
            
            # Mettre à jour le fichier de statut
            with open(status_file, "w") as f:
                f.write(f"=== STATUT D'EXÉCUTION NEURAX3 ===\n")
                f.write(f"Mise à jour: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Mode: TRAITEMENT COMPLET SANS LIMITATION\n")
                f.write(f"Progression globale: {report['global_progress']:.2f}%\n")
                f.write(f"Puzzles traités: {processed_puzzles}/{total_puzzles}\n")
                f.write(f"Dernier rapport: {report_text_filename}\n")
            
            # Attendre avant la prochaine génération de rapport
            time.sleep(600)  # 10 minutes
        except Exception as e:
            logger.error(f"Erreur dans la génération des rapports: {e}")
            time.sleep(60)  # En cas d'erreur, attendre 1 minute avant de réessayer

# Démarrer les threads de surveillance en arrière-plan
logger.info("Démarrage de la surveillance des checkpoints...")
checkpoint_thread = threading.Thread(target=monitor_checkpoints)
checkpoint_thread.daemon = True
checkpoint_thread.start()

logger.info("Démarrage de la génération des rapports d'avancement...")
report_thread = threading.Thread(target=generate_progress_reports)
report_thread.daemon = True
report_thread.start()

logger.info("Système de surveillance et d'analyse démarré avec succès")
logger.info("Vérifiez le dossier 'logs' pour les logs détaillés")
logger.info("Vérifiez le dossier 'reports' pour les rapports d'avancement")