# Script de surveillance Neurax3 pour Kaggle
# Ce script active la surveillance automatique de l'exécution

import os
import time
import threading
import json
import datetime
import glob
import logging
import sys

# Créer les dossiers nécessaires
os.makedirs("logs", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Configuration du logger
log_file = os.path.join("logs", f"neurax3_monitor_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("neurax3_monitor")

# Variables globales pour le suivi
puzzles_total = {
    "training": 1000,
    "evaluation": 120,
    "test": 240,
    "total": 1360
}

# Fonction de surveillance des checkpoints
def monitor_checkpoints():
    while True:
        try:
            checkpoint_files = glob.glob("*_checkpoint.json")
            if checkpoint_files:
                status = {"timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                
                for checkpoint_file in checkpoint_files:
                    phase = checkpoint_file.split("_")[0]
                    with open(checkpoint_file, 'r') as f:
                        data = json.load(f)
                    
                    processed_ids = data.get("processed_ids", [])
                    status[phase] = {
                        "processed": len(processed_ids),
                        "total": puzzles_total.get(phase, 0),
                        "progress": (len(processed_ids) / puzzles_total.get(phase, 1)) * 100
                    }
                    
                    logger.info(f"Phase {phase}: {len(processed_ids)}/{puzzles_total.get(phase, 0)} puzzles ({status[phase]['progress']:.2f}%)")
                
                # Écrire le statut dans un fichier
                with open(os.path.join("reports", f"status_{int(time.time())}.json"), 'w') as f:
                    json.dump(status, f, indent=2)
                
                # Mettre à jour le fichier de statut global
                with open(os.path.join("logs", "execution_status.txt"), 'w') as f:
                    f.write("=== STATUT D'EXÉCUTION NEURAX3 ===\n\n")
                    f.write(f"Date: {status['timestamp']}\n\n")
                    
                    # Calculer le progrès global
                    total_processed = sum(status.get(phase, {}).get("processed", 0) for phase in ["training", "evaluation", "test"])
                    global_progress = (total_processed / puzzles_total["total"]) * 100 if puzzles_total["total"] > 0 else 0
                    
                    f.write(f"Progression globale: {global_progress:.2f}%\n")
                    f.write(f"Puzzles traités: {total_processed}/{puzzles_total['total']}\n\n")
                    
                    for phase in ["training", "evaluation", "test"]:
                        if phase in status:
                            f.write(f"Phase {phase}:\n")
                            f.write(f"  Traités: {status[phase]['processed']}/{status[phase]['total']} ({status[phase]['progress']:.2f}%)\n")
                    
                    f.write("\nCe fichier est mis à jour automatiquement toutes les 5 minutes.")
            
            # Attendre avant la prochaine vérification
            time.sleep(300)  # 5 minutes
        except Exception as e:
            logger.error(f"Erreur dans la surveillance des checkpoints: {e}")
            time.sleep(60)  # Attendre 1 minute en cas d'erreur

# Fonction pour vérifier les erreurs
def check_for_errors():
    while True:
        try:
            # Vérifier s'il y a des exceptions non capturées dans les logs
            error_count = 0
            error_files = glob.glob("*.log") + glob.glob("logs/*.log")
            
            recent_errors = []
            for error_file in error_files:
                try:
                    with open(error_file, 'r') as f:
                        content = f.readlines()
                        for line in content[-100:]:  # Vérifier les 100 dernières lignes
                            if "ERROR" in line or "Exception" in line or "Error" in line:
                                error_count += 1
                                recent_errors.append(line.strip())
                                if len(recent_errors) >= 10:  # Limiter à 10 erreurs récentes
                                    break
                except Exception:
                    continue
            
            if error_count > 0:
                logger.warning(f"Détecté {error_count} erreurs dans les fichiers de log")
                
                # Enregistrer les erreurs récentes
                with open(os.path.join("logs", "recent_errors.txt"), 'w') as f:
                    f.write(f"=== ERREURS RÉCENTES ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===\n\n")
                    for i, error in enumerate(recent_errors):
                        f.write(f"{i+1}. {error}\n")
                    
                    f.write(f"\nTotal des erreurs détectées: {error_count}\n")
            
            # Attendre avant la prochaine vérification
            time.sleep(600)  # 10 minutes
        except Exception as e:
            logger.error(f"Erreur dans la vérification des erreurs: {e}")
            time.sleep(60)  # Attendre 1 minute en cas d'erreur

# Fonction pour créer des instantanés périodiques
def create_snapshots():
    while True:
        try:
            # Créer un timestamp
            timestamp = int(time.time())
            
            # Créer un dossier d'instantané
            snapshot_dir = os.path.join("reports", f"snapshot_{timestamp}")
            os.makedirs(snapshot_dir, exist_ok=True)
            
            # Copier les fichiers importants dans le dossier d'instantané
            for pattern in ["*_checkpoint.json", "*_results.json", "*_summary.json"]:
                for file in glob.glob(pattern):
                    try:
                        with open(file, 'r') as src_file, open(os.path.join(snapshot_dir, file), 'w') as dst_file:
                            content = src_file.read()
                            dst_file.write(content)
                    except Exception as e:
                        logger.error(f"Erreur lors de la copie de {file}: {e}")
            
            logger.info(f"Instantané créé: {snapshot_dir}")
            
            # Attendre avant le prochain instantané
            time.sleep(1800)  # 30 minutes
        except Exception as e:
            logger.error(f"Erreur lors de la création d'instantané: {e}")
            time.sleep(300)  # Attendre 5 minutes en cas d'erreur

# Démarrer les threads de surveillance
def start_monitoring():
    logger.info("=== DÉMARRAGE DU SYSTÈME DE SURVEILLANCE NEURAX3 ===")
    
    # Créer un fichier de statut initial
    with open(os.path.join("logs", "execution_status.txt"), 'w') as f:
        f.write("=== STATUT D'EXÉCUTION NEURAX3 ===\n\n")
        f.write(f"Démarrage du système de surveillance: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Le système de surveillance est en cours d'initialisation...\n")
        f.write("La première mise à jour aura lieu dans 5 minutes.\n")
    
    # Démarrer les threads
    checkpoint_thread = threading.Thread(target=monitor_checkpoints)
    checkpoint_thread.daemon = True
    checkpoint_thread.start()
    
    error_thread = threading.Thread(target=check_for_errors)
    error_thread.daemon = True
    error_thread.start()
    
    snapshot_thread = threading.Thread(target=create_snapshots)
    snapshot_thread.daemon = True
    snapshot_thread.start()
    
    logger.info("Tous les threads de surveillance ont été démarrés")
    logger.info("Le système de surveillance est maintenant actif")
    
    print("=== SYSTÈME DE SURVEILLANCE NEURAX3 ACTIVÉ ===")
    print("Les logs et rapports seront générés automatiquement.")
    print("Vérifiez le dossier 'logs' pour les fichiers de log détaillés.")
    print("Vérifiez le dossier 'reports' pour les rapports d'avancement.")
    print(f"Fichier de statut: logs/execution_status.txt")

# Démarrer la surveillance
start_monitoring()