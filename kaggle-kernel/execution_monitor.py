"""
Script de surveillance d'exécution pour Neurax3
Ce script surveille l'exécution du notebook Neurax3 et génère des rapports périodiques
pour valider que tout fonctionne correctement.
"""

import os
import json
import time
import logging
import datetime
import traceback
import threading
import glob

# Configuration du logger
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"execution_monitor_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('execution_monitor')

class ExecutionMonitor:
    """Classe pour surveiller l'exécution du notebook Neurax3"""
    
    def __init__(self, monitoring_interval=300):  # 5 minutes par défaut
        self.monitoring_interval = monitoring_interval
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        self.execution_stats = {
            "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_update": None,
            "monitoring_cycles": 0,
            "training": {"processed": 0, "total": 1000, "success_rate": 0},
            "evaluation": {"processed": 0, "total": 120, "success_rate": 0},
            "test": {"processed": 0, "total": 240, "success_rate": 0},
            "errors_detected": 0,
            "last_error": None
        }
    
    def start_monitoring(self):
        """Démarre la surveillance en arrière-plan"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("La surveillance est déjà en cours")
            return False
        
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info(f"Surveillance démarrée (intervalle: {self.monitoring_interval} secondes)")
        return True
    
    def stop_monitoring(self):
        """Arrête la surveillance"""
        if not self.monitoring_thread or not self.monitoring_thread.is_alive():
            logger.warning("Aucune surveillance en cours")
            return False
        
        self.stop_monitoring.set()
        self.monitoring_thread.join(timeout=10)
        
        logger.info("Surveillance arrêtée")
        return True
    
    def _monitoring_loop(self):
        """Boucle principale de surveillance"""
        while not self.stop_monitoring.is_set():
            try:
                self._check_execution_status()
                self._update_stats()
                self._generate_report()
                
                self.execution_stats["monitoring_cycles"] += 1
                self.execution_stats["last_update"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Enregistrer les statistiques
                with open(os.path.join(log_dir, "execution_stats.json"), "w") as f:
                    json.dump(self.execution_stats, f, indent=2)
            except Exception as e:
                logger.error(f"Erreur dans la boucle de surveillance: {e}")
                logger.error(traceback.format_exc())
            
            # Attendre l'intervalle de surveillance
            if not self.stop_monitoring.wait(timeout=self.monitoring_interval):
                continue
    
    def _check_execution_status(self):
        """Vérifie l'état d'exécution actuel"""
        logger.info("Vérification de l'état d'exécution...")
        
        # Vérifier les points de reprise
        for phase in ["training", "evaluation", "test"]:
            checkpoint_file = f"{phase}_checkpoint.json"
            if os.path.exists(checkpoint_file):
                try:
                    with open(checkpoint_file, "r") as f:
                        checkpoint = json.load(f)
                    
                    processed_ids = checkpoint.get("processed_ids", [])
                    self.execution_stats[phase]["processed"] = len(processed_ids)
                    
                    logger.info(f"Phase {phase}: {len(processed_ids)} puzzles traités")
                except Exception as e:
                    logger.error(f"Erreur lors de la lecture du fichier {checkpoint_file}: {e}")
        
        # Vérifier les résumés
        for phase in ["training", "evaluation", "test"]:
            summary_file = f"{phase}_summary.json"
            if os.path.exists(summary_file):
                try:
                    with open(summary_file, "r") as f:
                        summary = json.load(f)
                    
                    self.execution_stats[phase]["success_rate"] = summary.get("success_rate", 0)
                    
                    logger.info(f"Phase {phase}: Taux de réussite {summary.get('success_rate', 0):.2f}%")
                except Exception as e:
                    logger.error(f"Erreur lors de la lecture du fichier {summary_file}: {e}")
        
        # Vérifier les erreurs
        error_logs = glob.glob(os.path.join(log_dir, "error_*.log"))
        if error_logs:
            self.execution_stats["errors_detected"] = len(error_logs)
            
            # Lire la dernière erreur
            latest_error_log = max(error_logs, key=os.path.getctime)
            try:
                with open(latest_error_log, "r") as f:
                    last_lines = f.readlines()[-10:]  # Dernières 10 lignes
                    self.execution_stats["last_error"] = "".join(last_lines)
            except Exception as e:
                logger.error(f"Erreur lors de la lecture du fichier {latest_error_log}: {e}")
    
    def _update_stats(self):
        """Met à jour les statistiques d'exécution"""
        # Calculer la progression globale
        total_puzzles = sum(self.execution_stats[phase]["total"] for phase in ["training", "evaluation", "test"])
        processed_puzzles = sum(self.execution_stats[phase]["processed"] for phase in ["training", "evaluation", "test"])
        
        self.execution_stats["global_progress"] = {
            "total": total_puzzles,
            "processed": processed_puzzles,
            "percentage": (processed_puzzles / total_puzzles * 100) if total_puzzles > 0 else 0
        }
        
        # Calculer le temps écoulé
        start_time = datetime.datetime.strptime(self.execution_stats["start_time"], "%Y-%m-%d %H:%M:%S")
        elapsed = datetime.datetime.now() - start_time
        
        self.execution_stats["elapsed_time"] = {
            "seconds": elapsed.total_seconds(),
            "formatted": str(elapsed).split('.')[0]  # HH:MM:SS
        }
        
        # Estimer le temps restant
        if processed_puzzles > 0:
            seconds_per_puzzle = elapsed.total_seconds() / processed_puzzles
            remaining_puzzles = total_puzzles - processed_puzzles
            remaining_seconds = seconds_per_puzzle * remaining_puzzles
            
            remaining_time = datetime.timedelta(seconds=int(remaining_seconds))
            self.execution_stats["estimated_remaining"] = {
                "seconds": remaining_seconds,
                "formatted": str(remaining_time).split('.')[0]  # HH:MM:SS
            }
        else:
            self.execution_stats["estimated_remaining"] = {
                "seconds": 0,
                "formatted": "Inconnu"
            }
    
    def _generate_report(self):
        """Génère un rapport de surveillance"""
        report_path = os.path.join(log_dir, "execution_report.md")
        
        with open(report_path, "w") as f:
            f.write("# Rapport de surveillance d'exécution Neurax3\n\n")
            f.write(f"Date du rapport: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Progression globale\n\n")
            progress = self.execution_stats["global_progress"]["percentage"]
            progress_bar = "▓" * int(progress / 2) + "░" * (50 - int(progress / 2))
            f.write(f"```\n{progress_bar} {progress:.2f}%\n```\n\n")
            f.write(f"Puzzles traités: {self.execution_stats['global_progress']['processed']}/{self.execution_stats['global_progress']['total']}\n\n")
            
            f.write("## Temps d'exécution\n\n")
            f.write(f"Démarrage: {self.execution_stats['start_time']}\n")
            f.write(f"Temps écoulé: {self.execution_stats['elapsed_time']['formatted']}\n")
            f.write(f"Temps restant estimé: {self.execution_stats['estimated_remaining']['formatted']}\n\n")
            
            f.write("## Détails par phase\n\n")
            
            for phase in ["training", "evaluation", "test"]:
                f.write(f"### {phase.capitalize()}\n\n")
                progress = (self.execution_stats[phase]["processed"] / self.execution_stats[phase]["total"]) * 100
                progress_bar = "▓" * int(progress / 2) + "░" * (50 - int(progress / 2))
                f.write(f"```\n{progress_bar} {progress:.2f}%\n```\n\n")
                f.write(f"Puzzles traités: {self.execution_stats[phase]['processed']}/{self.execution_stats[phase]['total']}\n")
                f.write(f"Taux de réussite: {self.execution_stats[phase]['success_rate']:.2f}%\n\n")
            
            if self.execution_stats["errors_detected"] > 0:
                f.write("## Erreurs détectées\n\n")
                f.write(f"Nombre d'erreurs: {self.execution_stats['errors_detected']}\n\n")
                
                if self.execution_stats["last_error"]:
                    f.write("### Dernière erreur\n\n")
                    f.write("```\n")
                    f.write(self.execution_stats["last_error"])
                    f.write("```\n\n")
            
            f.write("## Statut\n\n")
            if self.execution_stats["global_progress"]["percentage"] >= 100:
                f.write("✅ EXÉCUTION TERMINÉE\n")
            elif self.execution_stats["errors_detected"] > 0:
                f.write("⚠️ EXÉCUTION EN COURS AVEC ERREURS\n")
            else:
                f.write("🔄 EXÉCUTION EN COURS\n")
        
        logger.info(f"Rapport de surveillance généré: {report_path}")

# Fonction principale
def main():
    """Fonction principale pour le moniteur d'exécution"""
    logger.info("=== DÉMARRAGE DU MONITEUR D'EXÉCUTION ===")
    
    # Créer et démarrer le moniteur
    monitor = ExecutionMonitor(monitoring_interval=300)  # 5 minutes
    monitor.start_monitoring()
    
    try:
        # Créer un fichier pour indiquer que le moniteur est actif
        status_path = os.path.join(log_dir, "monitor_status.txt")
        with open(status_path, "w") as f:
            f.write("=== STATUT DU MONITEUR D'EXÉCUTION ===\n")
            f.write(f"Démarré le: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Intervalle de surveillance: {monitor.monitoring_interval} secondes\n")
            f.write("Statut: ACTIF\n")
        
        logger.info(f"Fichier de statut créé: {status_path}")
        logger.info("Le moniteur d'exécution est actif et s'exécute en arrière-plan")
        
        # Pour le test, exécuter pendant une durée définie
        time.sleep(60)  # 1 minute pour le test
        
        # Dans une utilisation réelle, nous laisserions le thread daemon en arrière-plan
        # et le programme principal pourrait continuer
    except KeyboardInterrupt:
        logger.info("Interruption utilisateur, arrêt de la surveillance")
    except Exception as e:
        logger.error(f"Erreur non gérée: {e}")
        logger.error(traceback.format_exc())
    finally:
        # Mettre à jour le fichier de statut
        if os.path.exists(status_path):
            with open(status_path, "a") as f:
                f.write(f"Fin du moniteur: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        logger.info("=== FIN DU MONITEUR D'EXÉCUTION ===")

if __name__ == "__main__":
    main()