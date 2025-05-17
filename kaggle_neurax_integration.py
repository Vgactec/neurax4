#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script d'intégration Neurax2 avec l'API Kaggle pour la compétition ARC-Prize-2025
"""

import os
import sys
import json
import time
import logging
import tempfile
import argparse
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"kaggle_integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("KaggleIntegration")

# Identifiants API Kaggle
KAGGLE_USERNAME = "ndarray2000"
KAGGLE_API_KEY = "5354ea3f21950428c738b880332b0a5e"

class KaggleIntegration:
    """
    Classe pour l'intégration de Neurax2 avec Kaggle
    """
    
    def __init__(self, 
                competition_name: str = "arc-prize-2025",
                username: str = KAGGLE_USERNAME,
                api_key: str = KAGGLE_API_KEY,
                data_dir: str = "./kaggle_data",
                output_dir: str = "./kaggle_output"):
        """
        Initialise l'intégration Kaggle
        
        Args:
            competition_name: Nom de la compétition Kaggle
            username: Nom d'utilisateur Kaggle
            api_key: Clé API Kaggle
            data_dir: Répertoire pour les données téléchargées
            output_dir: Répertoire pour les fichiers de sortie
        """
        self.competition_name = competition_name
        self.username = username
        self.api_key = api_key
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Créer les répertoires
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Initialisation de l'intégration Kaggle pour la compétition {competition_name}")
        
        # Configurer l'API Kaggle
        self.configure_kaggle_api()
    
    def configure_kaggle_api(self) -> bool:
        """
        Configure l'API Kaggle avec les identifiants fournis
        
        Returns:
            True si la configuration a réussi, False sinon
        """
        try:
            # Créer le répertoire ~/.kaggle si nécessaire
            kaggle_dir = os.path.expanduser("~/.kaggle")
            os.makedirs(kaggle_dir, exist_ok=True)
            
            # Créer ou mettre à jour le fichier kaggle.json
            kaggle_json = {
                "username": self.username,
                "key": self.api_key
            }
            
            with open(os.path.join(kaggle_dir, "kaggle.json"), 'w') as f:
                json.dump(kaggle_json, f)
            
            # Restreindre les permissions
            os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)
            
            # Définir les variables d'environnement
            os.environ["KAGGLE_USERNAME"] = self.username
            os.environ["KAGGLE_KEY"] = self.api_key
            
            logger.info("Configuration de l'API Kaggle terminée")
            return True
        
        except Exception as e:
            logger.error(f"Erreur lors de la configuration de l'API Kaggle: {str(e)}")
            return False
    
    def install_kaggle_api(self) -> bool:
        """
        Installe la bibliothèque Kaggle-API si nécessaire
        
        Returns:
            True si l'installation a réussi, False sinon
        """
        try:
            # Vérifier si la bibliothèque est déjà installée
            try:
                import kaggle
                logger.info("Bibliothèque Kaggle-API déjà installée")
                return True
            except ImportError:
                # Installer la bibliothèque
                logger.info("Installation de la bibliothèque Kaggle-API")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "kaggle"],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    logger.error(f"Erreur lors de l'installation de Kaggle-API: {result.stderr}")
                    return False
                
                logger.info("Installation de Kaggle-API réussie")
                return True
        
        except Exception as e:
            logger.error(f"Erreur lors de l'installation de Kaggle-API: {str(e)}")
            return False
    
    def download_competition_data(self) -> bool:
        """
        Télécharge les données de la compétition
        
        Returns:
            True si le téléchargement a réussi, False sinon
        """
        try:
            # Installer la bibliothèque si nécessaire
            if not self.install_kaggle_api():
                logger.warning("Utilisation de la méthode alternative avec subprocess")
                # Méthode alternative avec subprocess
                cmd = f"kaggle competitions download -c {self.competition_name} -p {self.data_dir}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"Erreur lors du téléchargement: {result.stderr}")
                    return False
            else:
                # Utiliser la bibliothèque
                import kaggle
                api = kaggle.KaggleApi()
                api.authenticate()
                
                logger.info(f"Téléchargement des données pour la compétition {self.competition_name}")
                api.competition_download_files(self.competition_name, path=self.data_dir)
            
            # Extraction des fichiers ZIP
            for file in os.listdir(self.data_dir):
                if file.endswith('.zip'):
                    zip_path = os.path.join(self.data_dir, file)
                    extract_cmd = f"unzip -o '{zip_path}' -d {self.data_dir}"
                    subprocess.run(extract_cmd, shell=True, capture_output=True)
                    logger.info(f"Extraction de {file} terminée")
            
            logger.info("Téléchargement et extraction des données terminés")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du téléchargement des données: {str(e)}")
            return False
    
    def organize_data(self) -> Dict[str, str]:
        """
        Organise les données téléchargées dans la structure appropriée
        
        Returns:
            Dictionnaire avec les chemins vers les différents ensembles de données
        """
        try:
            # Créer les répertoires pour les différentes phases
            training_dir = os.path.join(self.data_dir, "training")
            evaluation_dir = os.path.join(self.data_dir, "evaluation")
            test_dir = os.path.join(self.data_dir, "test")
            
            os.makedirs(training_dir, exist_ok=True)
            os.makedirs(evaluation_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)
            
            # Répartir les fichiers selon leur type
            # Cette partie est spécifique à la structure de la compétition ARC-Prize-2025
            for file in os.listdir(self.data_dir):
                if file.endswith('.json'):
                    file_path = os.path.join(self.data_dir, file)
                    
                    # Déterminer la phase en fonction du nom du fichier
                    if "train" in file.lower():
                        # Traiter les données d'entraînement
                        self.process_dataset_file(file_path, training_dir)
                    elif "eval" in file.lower() or "valid" in file.lower():
                        # Traiter les données d'évaluation
                        self.process_dataset_file(file_path, evaluation_dir)
                    elif "test" in file.lower():
                        # Traiter les données de test
                        self.process_dataset_file(file_path, test_dir)
            
            logger.info(f"Organisation des données terminée: {len(os.listdir(training_dir))} puzzles d'entraînement, {len(os.listdir(evaluation_dir))} puzzles d'évaluation, {len(os.listdir(test_dir))} puzzles de test")
            
            return {
                "training": training_dir,
                "evaluation": evaluation_dir,
                "test": test_dir
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'organisation des données: {str(e)}")
            return {
                "training": "",
                "evaluation": "",
                "test": ""
            }
    
    def process_dataset_file(self, file_path: str, output_dir: str) -> None:
        """
        Traite un fichier de dataset et extrait les puzzles individuels
        
        Args:
            file_path: Chemin vers le fichier à traiter
            output_dir: Répertoire de sortie pour les puzzles individuels
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Déterminer le format du fichier
            if isinstance(data, dict):
                # Format où les clés sont les identifiants des puzzles
                for puzzle_id, puzzle_data in data.items():
                    puzzle_file = os.path.join(output_dir, f"{puzzle_id}.json")
                    with open(puzzle_file, 'w') as f:
                        json.dump(puzzle_data, f)
            elif isinstance(data, list):
                # Format où chaque élément est un puzzle avec un champ "id"
                for puzzle in data:
                    if "id" in puzzle:
                        puzzle_id = puzzle["id"]
                        puzzle_file = os.path.join(output_dir, f"{puzzle_id}.json")
                        with open(puzzle_file, 'w') as f:
                            json.dump(puzzle, f)
            elif "challenges" in data:
                # Format où les puzzles sont dans un champ "challenges"
                for puzzle in data["challenges"]:
                    if "id" in puzzle:
                        puzzle_id = puzzle["id"]
                        puzzle_file = os.path.join(output_dir, f"{puzzle_id}.json")
                        with open(puzzle_file, 'w') as f:
                            json.dump(puzzle, f)
        
        except Exception as e:
            logger.error(f"Erreur lors du traitement du fichier {file_path}: {str(e)}")
    
    def run_neurax_tests(self, data_paths: Dict[str, str], use_gpu: bool = True, use_mobile: bool = False) -> str:
        """
        Exécute les tests Neurax2 sur les données
        
        Args:
            data_paths: Chemins vers les différents ensembles de données
            use_gpu: Utiliser le GPU si disponible
            use_mobile: Utiliser la version optimisée pour mobile
            
        Returns:
            Chemin vers le fichier de soumission
        """
        try:
            # Construire la commande
            cmd = [
                sys.executable,
                "run_complete_arc_test.py",
                "--all",
                "--gpu" if use_gpu else ""
            ]
            # Filtrer les arguments vides
            cmd = [arg for arg in cmd if arg]
            
            # Ajouter les options
            if not use_gpu:
                cmd.append("--no-gpu")
            
            if use_mobile:
                cmd.append("--mobile")
            
            # Exécuter la commande
            logger.info(f"Exécution de la commande: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Erreur lors de l'exécution des tests: {result.stderr}")
                return ""
            
            # Trouver le fichier de rapport le plus récent
            report_files = [f for f in os.listdir(self.output_dir) if f.startswith("arc_batch_test_report_") and f.endswith(".md")]
            if report_files:
                report_files.sort(reverse=True)  # Le plus récent en premier
                report_path = os.path.join(self.output_dir, report_files[0])
                logger.info(f"Rapport généré: {report_path}")
            
            # Trouver le fichier de résultats de test le plus récent
            test_files = [f for f in os.listdir(self.output_dir) if f.startswith("arc_batch_test_test_") and f.endswith(".json")]
            if test_files:
                test_files.sort(reverse=True)  # Le plus récent en premier
                test_results_path = os.path.join(self.output_dir, test_files[0])
                
                # Créer un fichier de soumission à partir des résultats de test
                submission_path = os.path.join(self.output_dir, "submission.csv")
                self.create_submission_file(test_results_path, submission_path)
                
                return submission_path
            
            return ""
            
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution des tests: {str(e)}")
            return ""
    
    def create_submission_file(self, test_results_path: str, submission_path: str) -> bool:
        """
        Crée un fichier de soumission à partir des résultats de test
        
        Args:
            test_results_path: Chemin vers le fichier de résultats de test
            submission_path: Chemin vers le fichier de soumission à créer
            
        Returns:
            True si la création a réussi, False sinon
        """
        try:
            # Charger les résultats de test
            with open(test_results_path, 'r') as f:
                test_results = json.load(f)
            
            # Créer le fichier de soumission au format CSV
            with open(submission_path, 'w', newline='') as f:
                f.write("puzzle_id,output\n")  # En-tête CSV
                
                for result in test_results:
                    puzzle_id = result.get("puzzle_id", "")
                    prediction = result.get("prediction", [])
                    
                    # Convertir la prédiction en JSON
                    prediction_json = json.dumps(prediction)
                    
                    # Écrire la ligne
                    f.write(f"{puzzle_id},{prediction_json}\n")
            
            logger.info(f"Fichier de soumission créé: {submission_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la création du fichier de soumission: {str(e)}")
            return False
    
    def submit_to_kaggle(self, submission_path: str, message: str = "Neurax2 Submission") -> bool:
        """
        Soumet le fichier de résultats à Kaggle
        
        Args:
            submission_path: Chemin vers le fichier de soumission
            message: Message pour la soumission
            
        Returns:
            True si la soumission a réussi, False sinon
        """
        try:
            # Vérifier que le fichier existe
            if not os.path.exists(submission_path):
                logger.error(f"Le fichier de soumission {submission_path} n'existe pas")
                return False
            
            # Installer la bibliothèque si nécessaire
            if not self.install_kaggle_api():
                logger.warning("Utilisation de la méthode alternative avec subprocess")
                # Méthode alternative avec subprocess
                cmd = f"kaggle competitions submit -c {self.competition_name} -f '{submission_path}' -m '{message}'"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"Erreur lors de la soumission: {result.stderr}")
                    return False
            else:
                # Utiliser la bibliothèque
                import kaggle
                api = kaggle.KaggleApi()
                api.authenticate()
                
                logger.info(f"Soumission du fichier {submission_path} à la compétition {self.competition_name}")
                api.competition_submit(submission_path, message, self.competition_name)
            
            logger.info("Soumission réussie")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la soumission: {str(e)}")
            return False
    
    def run_complete_workflow(self, use_gpu: bool = True, use_mobile: bool = False) -> bool:
        """
        Exécute le workflow complet: téléchargement, organisation, exécution des tests et soumission
        
        Args:
            use_gpu: Utiliser le GPU si disponible
            use_mobile: Utiliser la version optimisée pour mobile
            
        Returns:
            True si le workflow a réussi, False sinon
        """
        # Télécharger les données
        if not self.download_competition_data():
            logger.error("Échec du téléchargement des données")
            return False
        
        # Organiser les données
        data_paths = self.organize_data()
        if not all(data_paths.values()):
            logger.error("Échec de l'organisation des données")
            return False
        
        # Exécuter les tests
        submission_path = self.run_neurax_tests(data_paths, use_gpu, use_mobile)
        if not submission_path:
            logger.error("Échec de l'exécution des tests")
            return False
        
        # Soumettre les résultats
        config_str = f"GPU={'Activé' if use_gpu else 'Désactivé'}, Mobile={'Activé' if use_mobile else 'Désactivé'}"
        message = f"Neurax2 Submission - {config_str} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        if not self.submit_to_kaggle(submission_path, message):
            logger.error("Échec de la soumission")
            return False
        
        logger.info("Workflow complet terminé avec succès")
        return True


def main():
    """
    Point d'entrée principal
    """
    parser = argparse.ArgumentParser(description="Intégration Neurax2 avec Kaggle pour ARC-Prize-2025")
    parser.add_argument("--competition", type=str, default="arc-prize-2025",
                      help="Nom de la compétition Kaggle")
    parser.add_argument("--username", type=str, default=KAGGLE_USERNAME,
                      help="Nom d'utilisateur Kaggle")
    parser.add_argument("--api-key", type=str, default=KAGGLE_API_KEY,
                      help="Clé API Kaggle")
    parser.add_argument("--data-dir", type=str, default="./kaggle_data",
                      help="Répertoire pour les données téléchargées")
    parser.add_argument("--output-dir", type=str, default="./kaggle_output",
                      help="Répertoire pour les fichiers de sortie")
    parser.add_argument("--no-gpu", action="store_true",
                      help="Désactiver l'utilisation du GPU")
    parser.add_argument("--mobile", action="store_true",
                      help="Utiliser la version optimisée pour mobile")
    parser.add_argument("--download-only", action="store_true",
                      help="Télécharger uniquement les données")
    parser.add_argument("--test-only", action="store_true",
                      help="Exécuter uniquement les tests sans soumettre")
    parser.add_argument("--submit-only", type=str, default="",
                      help="Soumettre uniquement le fichier spécifié")
    
    args = parser.parse_args()
    
    # Créer l'intégration
    integration = KaggleIntegration(
        competition_name=args.competition,
        username=args.username,
        api_key=args.api_key,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # Exécuter l'action spécifiée
    if args.download_only:
        integration.download_competition_data()
        integration.organize_data()
    elif args.test_only:
        data_paths = integration.organize_data()
        integration.run_neurax_tests(data_paths, not args.no_gpu, args.mobile)
    elif args.submit_only:
        integration.submit_to_kaggle(args.submit_only)
    else:
        integration.run_complete_workflow(not args.no_gpu, args.mobile)


if __name__ == "__main__":
    main()