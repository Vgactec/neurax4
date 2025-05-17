#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module de vérification complète du système Neurax2
Contrôle point par point tous les prérequis et assure l'exécution sans erreur
"""

import os
import sys
import json
import time
import logging
import subprocess
import platform
import importlib
import shutil
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"neurax_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("NeuraxVerification")

class SystemVerifier:
    """
    Vérificateur complet du système Neurax2
    """
    
    def __init__(self):
        """
        Initialise le vérificateur
        """
        self.system_info = self._get_system_info()
        self.verification_results = {}
        self.all_scripts = []
        self.all_logs = []
        self.all_required_modules = [
            "numpy", "matplotlib", "pandas", "scipy", 
            "json", "time", "logging", "subprocess", 
            "glob", "os", "sys", "datetime", "random"
        ]
        self.required_files = [
            "run_complete_arc_benchmarks.sh",
            "run_complete_arc_test.py",
            "run_learning_analysis.py",
            "optimize_learning_rate.py",
            "run_complete_pipeline.py",
            "visualize_complete_results.py",
            "neurax_engine.py"
        ]
        
        logger.info(f"Vérificateur système initialisé sur {self.system_info['platform']}")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """
        Récupère les informations système
        
        Returns:
            Informations sur le système
        """
        info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "memory": self._get_memory_info(),
            "disk_space": self._get_disk_space()
        }
        
        return info
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """
        Récupère les informations sur la mémoire
        
        Returns:
            Informations sur la mémoire
        """
        try:
            import psutil
            vm = psutil.virtual_memory()
            return {
                "total": vm.total / (1024**3),  # GB
                "available": vm.available / (1024**3),  # GB
                "used": vm.used / (1024**3),  # GB
                "percent": vm.percent
            }
        except ImportError:
            return {"error": "psutil non disponible"}
    
    def _get_disk_space(self) -> Dict[str, Any]:
        """
        Récupère les informations sur l'espace disque
        
        Returns:
            Informations sur l'espace disque
        """
        try:
            import psutil
            disk = psutil.disk_usage('/')
            return {
                "total": disk.total / (1024**3),  # GB
                "used": disk.used / (1024**3),  # GB
                "free": disk.free / (1024**3),  # GB
                "percent": disk.percent
            }
        except ImportError:
            return {"error": "psutil non disponible"}
    
    def verify_python_modules(self) -> bool:
        """
        Vérifie la disponibilité des modules Python requis
        
        Returns:
            True si tous les modules sont disponibles, False sinon
        """
        logger.info("Vérification des modules Python requis")
        
        module_status = {}
        all_available = True
        
        for module_name in self.all_required_modules:
            try:
                importlib.import_module(module_name)
                module_status[module_name] = "disponible"
                logger.info(f"Module {module_name} disponible")
            except ImportError:
                module_status[module_name] = "non disponible"
                all_available = False
                logger.error(f"Module {module_name} non disponible")
        
        try:
            import cupy
            has_gpu = True
            module_status["cupy"] = "disponible"
            logger.info("Module cupy disponible (support GPU)")
        except ImportError:
            has_gpu = False
            module_status["cupy"] = "non disponible"
            logger.warning("Module cupy non disponible (pas de support GPU)")
        
        self.verification_results["modules"] = {
            "status": all_available,
            "details": module_status,
            "has_gpu": has_gpu
        }
        
        return all_available
    
    def verify_required_files(self) -> bool:
        """
        Vérifie la présence des fichiers requis
        
        Returns:
            True si tous les fichiers sont présents, False sinon
        """
        logger.info("Vérification des fichiers requis")
        
        file_status = {}
        all_present = True
        
        for file_name in self.required_files:
            if os.path.exists(file_name):
                file_status[file_name] = {
                    "exists": True,
                    "size": os.path.getsize(file_name),
                    "executable": os.access(file_name, os.X_OK) if file_name.endswith(".py") or file_name.endswith(".sh") else None
                }
                logger.info(f"Fichier {file_name} présent ({file_status[file_name]['size']} octets)")
            else:
                file_status[file_name] = {"exists": False}
                all_present = False
                logger.error(f"Fichier {file_name} non trouvé")
        
        self.verification_results["files"] = {
            "status": all_present,
            "details": file_status
        }
        
        return all_present
    
    def verify_execution_permissions(self) -> bool:
        """
        Vérifie et corrige les permissions d'exécution des scripts
        
        Returns:
            True si toutes les permissions sont correctes, False sinon
        """
        logger.info("Vérification des permissions d'exécution")
        
        # Identifier tous les scripts
        scripts = [f for f in os.listdir(".") if f.endswith(".py") or f.endswith(".sh")]
        self.all_scripts = scripts
        
        permission_status = {}
        all_correct = True
        
        for script in scripts:
            if os.access(script, os.X_OK):
                permission_status[script] = "exécutable"
                logger.info(f"Script {script} déjà exécutable")
            else:
                try:
                    os.chmod(script, 0o755)
                    permission_status[script] = "corrigé"
                    logger.info(f"Permissions corrigées pour {script}")
                except Exception as e:
                    permission_status[script] = f"erreur: {str(e)}"
                    all_correct = False
                    logger.error(f"Impossible de corriger les permissions pour {script}: {str(e)}")
        
        self.verification_results["permissions"] = {
            "status": all_correct,
            "details": permission_status
        }
        
        return all_correct
    
    def ensure_log_directories(self) -> bool:
        """
        Vérifie et crée les répertoires pour les logs
        
        Returns:
            True si les répertoires sont prêts, False sinon
        """
        logger.info("Vérification des répertoires de logs")
        
        log_dirs = ["logs", "arc_results", "output"]
        dir_status = {}
        all_ready = True
        
        for log_dir in log_dirs:
            if os.path.exists(log_dir):
                if os.path.isdir(log_dir):
                    dir_status[log_dir] = "existant"
                    logger.info(f"Répertoire {log_dir} existant")
                else:
                    try:
                        os.remove(log_dir)
                        os.mkdir(log_dir)
                        dir_status[log_dir] = "recréé"
                        logger.info(f"Fichier {log_dir} remplacé par un répertoire")
                    except Exception as e:
                        dir_status[log_dir] = f"erreur: {str(e)}"
                        all_ready = False
                        logger.error(f"Impossible de recréer le répertoire {log_dir}: {str(e)}")
            else:
                try:
                    os.mkdir(log_dir)
                    dir_status[log_dir] = "créé"
                    logger.info(f"Répertoire {log_dir} créé")
                except Exception as e:
                    dir_status[log_dir] = f"erreur: {str(e)}"
                    all_ready = False
                    logger.error(f"Impossible de créer le répertoire {log_dir}: {str(e)}")
        
        self.verification_results["log_dirs"] = {
            "status": all_ready,
            "details": dir_status
        }
        
        return all_ready
    
    def check_benchmark_script(self) -> bool:
        """
        Vérifie et corrige le script de benchmark
        
        Returns:
            True si le script est valide, False sinon
        """
        logger.info("Vérification du script de benchmark")
        
        benchmark_script = "run_complete_arc_benchmarks.sh"
        if not os.path.exists(benchmark_script):
            logger.error(f"Script {benchmark_script} non trouvé")
            return False
        
        # Lire le contenu du script
        try:
            with open(benchmark_script, 'r') as f:
                content = f.read()
            
            # Vérifier les points critiques
            checks = {
                "redirections": "> benchmark_progress.log" in content,
                "error_handling": "if [ $? -ne 0 ]" in content,
                "executable": os.access(benchmark_script, os.X_OK)
            }
            
            all_valid = all(checks.values())
            
            # Corriger les problèmes si nécessaire
            if not checks["redirections"]:
                logger.warning("Redirection des logs manquante dans le script de benchmark")
                # Ajouter la redirection si nécessaire
            
            if not checks["executable"]:
                os.chmod(benchmark_script, 0o755)
                logger.info(f"Permissions corrigées pour {benchmark_script}")
            
            self.verification_results["benchmark_script"] = {
                "status": all_valid,
                "details": checks
            }
            
            return all_valid
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification du script de benchmark: {str(e)}")
            return False
    
    def create_test_benchmark_log(self) -> bool:
        """
        Crée un fichier de log de test pour vérifier les permissions
        
        Returns:
            True si le fichier a été créé, False sinon
        """
        logger.info("Création d'un fichier de log de test")
        
        try:
            with open("benchmark_progress.log", 'w') as f:
                f.write(f"# Log de test généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("# Ce fichier vérifie les permissions d'écriture\n")
                f.write("Verification du processus:\n")
                f.write("[INFO] Démarrage du benchmark complet de Neurax2...\n")
            
            if os.path.exists("benchmark_progress.log"):
                logger.info("Fichier de log de test créé avec succès")
                return True
            else:
                logger.error("Échec de la création du fichier de log de test")
                return False
                
        except Exception as e:
            logger.error(f"Erreur lors de la création du fichier de log de test: {str(e)}")
            return False
    
    def detect_existing_logs(self) -> List[str]:
        """
        Détecte les fichiers de log existants
        
        Returns:
            Liste des fichiers de log trouvés
        """
        logger.info("Détection des fichiers de log existants")
        
        log_patterns = [
            "*.log", 
            "arc_*.csv", 
            "arc_*.json",
            "neurax_*.log", 
            "benchmark_*.log",
            "*/training_*.json",
            "*/evaluation_*.json",
            "*/test_*.json"
        ]
        
        found_logs = []
        
        for pattern in log_patterns:
            import glob
            logs = glob.glob(pattern)
            found_logs.extend(logs)
        
        self.all_logs = found_logs
        logger.info(f"Trouvé {len(found_logs)} fichiers de log")
        
        return found_logs
    
    def fix_run_script(self) -> bool:
        """
        Corrige le script d'exécution principal pour garantir la création des logs
        
        Returns:
            True si le script a été corrigé, False sinon
        """
        logger.info("Correction du script d'exécution principal")
        
        script_path = "run_complete_arc_benchmarks.sh"
        if not os.path.exists(script_path):
            logger.error(f"Script {script_path} non trouvé")
            return False
        
        # Lire le contenu actuel
        try:
            with open(script_path, 'r') as f:
                content = f.readlines()
            
            # Chercher et corriger les problèmes
            modified = False
            new_content = []
            
            for line in content:
                # Ajouter des vérifications de log
                if "function main()" in line:
                    new_content.append(line)
                    new_content.append("    # Créer le fichier de log s'il n'existe pas\n")
                    new_content.append("    touch benchmark_progress.log\n")
                    new_content.append("    chmod 666 benchmark_progress.log\n")
                    new_content.append("    echo \"# Log démarré le $(date)\" > benchmark_progress.log\n\n")
                    modified = True
                # Améliorer l'écriture dans les logs
                elif "log_info" in line and ">>>" not in line:
                    new_line = line.replace("log_info", "log_info").replace("\n", " | tee -a benchmark_progress.log\n")
                    new_content.append(new_line)
                    modified = True
                else:
                    new_content.append(line)
            
            # Écrire le contenu modifié
            if modified:
                with open(script_path, 'w') as f:
                    f.writelines(new_content)
                logger.info(f"Script {script_path} corrigé avec succès")
                return True
            else:
                logger.info(f"Script {script_path} déjà correct")
                return True
                
        except Exception as e:
            logger.error(f"Erreur lors de la correction du script: {str(e)}")
            return False
    
    def run_verification(self) -> Dict[str, Any]:
        """
        Exécute toutes les vérifications
        
        Returns:
            Résultats des vérifications
        """
        logger.info("Démarrage de la vérification complète du système")
        
        # Vérifications principales
        modules_ok = self.verify_python_modules()
        files_ok = self.verify_required_files()
        permissions_ok = self.verify_execution_permissions()
        log_dirs_ok = self.ensure_log_directories()
        benchmark_script_ok = self.check_benchmark_script()
        
        # Création du fichier de log de test
        test_log_ok = self.create_test_benchmark_log()
        
        # Détection des logs existants
        existing_logs = self.detect_existing_logs()
        
        # Correction du script d'exécution
        script_fixed = self.fix_run_script()
        
        # Résumé global
        all_ok = modules_ok and files_ok and permissions_ok and log_dirs_ok and benchmark_script_ok and test_log_ok and script_fixed
        
        self.verification_results["summary"] = {
            "status": all_ok,
            "timestamp": datetime.now().isoformat(),
            "system_info": self.system_info,
            "modules_ok": modules_ok,
            "files_ok": files_ok,
            "permissions_ok": permissions_ok,
            "log_dirs_ok": log_dirs_ok,
            "benchmark_script_ok": benchmark_script_ok,
            "test_log_ok": test_log_ok,
            "script_fixed": script_fixed,
            "existing_logs": len(existing_logs)
        }
        
        logger.info(f"Vérification complète terminée - Résultat: {'OK' if all_ok else 'ÉCHEC'}")
        
        return self.verification_results
    
    def generate_report(self) -> str:
        """
        Génère un rapport de vérification
        
        Returns:
            Rapport de vérification au format Markdown
        """
        if not self.verification_results:
            self.run_verification()
        
        summary = self.verification_results.get("summary", {})
        
        report = f"""# Rapport de Vérification du Système Neurax2

## Résumé

- **État global**: {'✅ OK' if summary.get('status', False) else '❌ ÉCHEC'}
- **Date de vérification**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Plateforme**: {self.system_info.get('platform', 'Inconnue')}
- **Version Python**: {self.system_info.get('python_version', 'Inconnue')}

## Vérifications Détaillées

| Vérification | Résultat | Détails |
|--------------|----------|---------|
| Modules Python | {'✅ OK' if summary.get('modules_ok', False) else '❌ ÉCHEC'} | {f"{len(self.all_required_modules)} modules requis, GPU: {'disponible' if self.verification_results.get('modules', {}).get('details', {}).get('has_gpu', False) else 'non disponible'}"} |
| Fichiers requis | {'✅ OK' if summary.get('files_ok', False) else '❌ ÉCHEC'} | {f"{len(self.required_files)} fichiers nécessaires"} |
| Permissions d'exécution | {'✅ OK' if summary.get('permissions_ok', False) else '❌ ÉCHEC'} | {f"{len(self.all_scripts)} scripts vérifiés"} |
| Répertoires de logs | {'✅ OK' if summary.get('log_dirs_ok', False) else '❌ ÉCHEC'} | Répertoires créés et accessibles |
| Script de benchmark | {'✅ OK' if summary.get('benchmark_script_ok', False) else '❌ ÉCHEC'} | Redirection de logs et gestion d'erreurs |
| Test de log | {'✅ OK' if summary.get('test_log_ok', False) else '❌ ÉCHEC'} | Création du fichier de log test |
| Script corrigé | {'✅ OK' if summary.get('script_fixed', False) else '❌ ÉCHEC'} | Améliorations du script de benchmark |

## Ressources Système

- **Mémoire**: {self.system_info.get('memory', {}).get('total', 'N/A'):.2f} GB total, {self.system_info.get('memory', {}).get('available', 'N/A'):.2f} GB disponible
- **Espace disque**: {self.system_info.get('disk_space', {}).get('total', 'N/A'):.2f} GB total, {self.system_info.get('disk_space', {}).get('free', 'N/A'):.2f} GB libre

## Fichiers de Logs Existants

{len(self.all_logs)} fichiers de logs trouvés dans le système.

## Recommandations

{
"Tous les prérequis sont satisfaits. Le système est prêt à exécuter les benchmarks complets." 
if summary.get('status', False) else 
"Des corrections sont nécessaires avant de lancer les benchmarks complets. Consultez les détails ci-dessus."
}

---

*Rapport généré le {datetime.now().strftime('%Y-%m-%d à %H:%M:%S')}*
"""
        
        # Enregistrer le rapport
        report_file = f"rapport_verification_systeme_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Rapport de vérification enregistré: {report_file}")
        
        return report
    
    def fix_all_issues(self) -> bool:
        """
        Corrige automatiquement tous les problèmes identifiés
        
        Returns:
            True si toutes les corrections ont réussi, False sinon
        """
        logger.info("Correction automatique de tous les problèmes")
        
        # Exécuter les vérifications si pas déjà fait
        if not self.verification_results:
            self.run_verification()
        
        # Corriger les permissions de tous les scripts
        for script in self.all_scripts:
            try:
                os.chmod(script, 0o755)
                logger.info(f"Permissions corrigées pour {script}")
            except Exception as e:
                logger.error(f"Impossible de corriger les permissions pour {script}: {str(e)}")
        
        # Créer tous les répertoires de logs
        for log_dir in ["logs", "arc_results", "output"]:
            if not os.path.exists(log_dir):
                try:
                    os.mkdir(log_dir)
                    logger.info(f"Répertoire {log_dir} créé")
                except Exception as e:
                    logger.error(f"Impossible de créer le répertoire {log_dir}: {str(e)}")
        
        # Créer un fichier de log vide
        try:
            with open("benchmark_progress.log", 'w') as f:
                f.write(f"# Log de benchmark créé le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("# Ce fichier a été créé par le système de vérification\n")
            logger.info("Fichier de log benchmark_progress.log créé")
        except Exception as e:
            logger.error(f"Impossible de créer le fichier de log: {str(e)}")
        
        # Corriger le script de benchmark
        self.fix_run_script()
        
        logger.info("Correction automatique terminée")
        
        # Vérifier à nouveau pour confirmer les corrections
        return self.run_verification()["summary"]["status"]


def main():
    """
    Point d'entrée principal
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Vérification complète du système Neurax2")
    parser.add_argument("--fix", action="store_true",
                      help="Corriger automatiquement les problèmes détectés")
    parser.add_argument("--report", action="store_true",
                      help="Générer un rapport de vérification")
    
    args = parser.parse_args()
    
    verifier = SystemVerifier()
    
    if args.fix:
        verifier.fix_all_issues()
    else:
        verifier.run_verification()
    
    if args.report:
        verifier.generate_report()


if __name__ == "__main__":
    main()