#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script d'intégration Neurax2 pour Kaggle ARC-Prize-2025
Utilise l'API Kaggle pour télécharger les données et soumettre les résultats
"""

import os
import sys
import json
import logging
import subprocess
import argparse
from typing import Dict, List, Any, Optional

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("neurax_kaggle.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("NeuraxKaggle")

# Identifiants Kaggle (utilisez les secrets configurés)
KAGGLE_USERNAME = "ndarray2000"
KAGGLE_KEY = "5354ea3f21950428c738b880332b0a5e"  # Clé d'API Kaggle

class KaggleIntegration:
    """
    Classe pour l'intégration de Neurax2 avec Kaggle pour la compétition ARC-Prize-2025
    """
    
    def __init__(self, 
                competition_name: str = "arc-prize-2025",
                data_dir: str = "./kaggle_data",
                output_dir: str = "./kaggle_output",
                username: Optional[str] = None,
                api_key: Optional[str] = None):
        """
        Initialise l'intégration Kaggle
        
        Args:
            competition_name: Nom de la compétition Kaggle
            data_dir: Répertoire pour les données téléchargées
            output_dir: Répertoire pour les fichiers de soumission
            username: Nom d'utilisateur Kaggle (facultatif, utilise KAGGLE_USERNAME par défaut)
            api_key: Clé API Kaggle (facultatif, utilise KAGGLE_KEY par défaut)
        """
        self.competition_name = competition_name
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.username = username or KAGGLE_USERNAME
        self.api_key = api_key or KAGGLE_KEY
        
        # Créer les répertoires nécessaires
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialiser l'API Kaggle
        self.setup_kaggle_api()
    
    def setup_kaggle_api(self) -> bool:
        """
        Configure l'API Kaggle avec les identifiants fournis
        
        Returns:
            True si la configuration a réussi, False sinon
        """
        try:
            # Créer le répertoire Kaggle si nécessaire
            kaggle_dir = os.path.expanduser("~/.kaggle")
            os.makedirs(kaggle_dir, exist_ok=True)
            
            # Créer le fichier de configuration
            kaggle_config = {
                "username": self.username,
                "key": self.api_key
            }
            
            with open(os.path.join(kaggle_dir, "kaggle.json"), "w") as f:
                json.dump(kaggle_config, f)
            
            # Restreindre les permissions
            try:
                os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)
            except Exception as e:
                logger.warning(f"Impossible de modifier les permissions du fichier de configuration: {str(e)}")
            
            # Définir les variables d'environnement
            os.environ["KAGGLE_USERNAME"] = self.username
            os.environ["KAGGLE_KEY"] = self.api_key
            
            logger.info("Configuration de l'API Kaggle réussie")
            return True
        
        except Exception as e:
            logger.error(f"Erreur lors de la configuration de l'API Kaggle: {str(e)}")
            return False
    
    def download_competition_data(self) -> bool:
        """
        Télécharge les données de la compétition
        
        Returns:
            True si le téléchargement a réussi, False sinon
        """
        try:
            # Vérifier si l'API Kaggle est installée
            try:
                import kaggle
                api = kaggle.KaggleApi()
                api.authenticate()
                
                logger.info(f"Téléchargement des données pour la compétition {self.competition_name}")
                api.competition_download_files(self.competition_name, path=self.data_dir)
                
                # Extraire les fichiers ZIP
                for file in os.listdir(self.data_dir):
                    if file.endswith('.zip'):
                        import zipfile
                        with zipfile.ZipFile(os.path.join(self.data_dir, file), 'r') as zip_ref:
                            zip_ref.extractall(self.data_dir)
                
                logger.info("Téléchargement et extraction des données terminés")
                return True
                
            except ImportError:
                # Fallback sur la commande shell si la bibliothèque n'est pas installée
                logger.warning("Module kaggle non trouvé, utilisation de la commande shell")
                cmd = f"kaggle competitions download -c {self.competition_name} -p {self.data_dir}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"Erreur lors du téléchargement: {result.stderr}")
                    return False
                
                # Extraire les fichiers ZIP
                cmd = f"unzip -o '{self.data_dir}/*.zip' -d {self.data_dir}"
                subprocess.run(cmd, shell=True)
                
                logger.info("Téléchargement et extraction des données terminés")
                return True
        
        except Exception as e:
            logger.error(f"Erreur lors du téléchargement des données: {str(e)}")
            return False
    
    def organize_data_directories(self) -> bool:
        """
        Organise les données téléchargées dans la structure attendue par Neurax2
        
        Returns:
            True si l'organisation a réussi, False sinon
        """
        try:
            # Créer les répertoires pour les différentes phases
            training_dir = os.path.join(self.data_dir, "training")
            evaluation_dir = os.path.join(self.data_dir, "evaluation")
            test_dir = os.path.join(self.data_dir, "test")
            
            os.makedirs(training_dir, exist_ok=True)
            os.makedirs(evaluation_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)
            
            # Analyser les fichiers téléchargés et les répartir
            # Cette partie peut varier selon la structure exacte des données de la compétition
            # Implémentation préliminaire basée sur les formats typiques
            
            for file in os.listdir(self.data_dir):
                if file.endswith('.json'):
                    if 'train' in file.lower():
                        # Traiter les données d'entraînement
                        with open(os.path.join(self.data_dir, file), 'r') as f:
                            data = json.load(f)
                        
                        # Sauvegarder chaque puzzle individuellement
                        for puzzle_id, puzzle_data in data.items():
                            with open(os.path.join(training_dir, f"{puzzle_id}.json"), 'w') as f:
                                json.dump(puzzle_data, f)
                    
                    elif 'eval' in file.lower() or 'valid' in file.lower():
                        # Traiter les données d'évaluation
                        with open(os.path.join(self.data_dir, file), 'r') as f:
                            data = json.load(f)
                        
                        for puzzle_id, puzzle_data in data.items():
                            with open(os.path.join(evaluation_dir, f"{puzzle_id}.json"), 'w') as f:
                                json.dump(puzzle_data, f)
                    
                    elif 'test' in file.lower():
                        # Traiter les données de test
                        with open(os.path.join(self.data_dir, file), 'r') as f:
                            data = json.load(f)
                        
                        for puzzle_id, puzzle_data in data.items():
                            with open(os.path.join(test_dir, f"{puzzle_id}.json"), 'w') as f:
                                json.dump(puzzle_data, f)
            
            logger.info("Organisation des données terminée")
            return True
        
        except Exception as e:
            logger.error(f"Erreur lors de l'organisation des données: {str(e)}")
            return False
    
    def submit_results(self, submission_file: str, message: str = "Automatic submission") -> bool:
        """
        Soumet les résultats à Kaggle
        
        Args:
            submission_file: Chemin vers le fichier de soumission
            message: Message pour la soumission
            
        Returns:
            True si la soumission a réussi, False sinon
        """
        try:
            # Vérifier si l'API Kaggle est installée
            try:
                import kaggle
                api = kaggle.KaggleApi()
                api.authenticate()
                
                logger.info(f"Soumission des résultats pour la compétition {self.competition_name}")
                api.competition_submit(submission_file, message, self.competition_name)
                
                logger.info("Soumission des résultats réussie")
                return True
                
            except ImportError:
                # Fallback sur la commande shell si la bibliothèque n'est pas installée
                logger.warning("Module kaggle non trouvé, utilisation de la commande shell")
                cmd = f'kaggle competitions submit -c {self.competition_name} -f "{submission_file}" -m "{message}"'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"Erreur lors de la soumission: {result.stderr}")
                    return False
                
                logger.info("Soumission des résultats réussie")
                return True
        
        except Exception as e:
            logger.error(f"Erreur lors de la soumission des résultats: {str(e)}")
            return False
    
    def run_neurax_processing(self, use_gpu: bool = True, use_mobile: bool = False) -> str:
        """
        Exécute le traitement Neurax2 sur les données
        
        Args:
            use_gpu: Utiliser le GPU si disponible
            use_mobile: Utiliser la version mobile du simulateur
            
        Returns:
            Chemin vers le fichier de soumission généré
        """
        try:
            # Chemin du script d'adaptateur
            script_path = "kaggle_arc_prize_2025.py"
            
            # Construire la commande
            cmd = [
                "python", script_path,
                "--training", os.path.join(self.data_dir, "training"),
                "--evaluation", os.path.join(self.data_dir, "evaluation"),
                "--test", os.path.join(self.data_dir, "test"),
                "--output", self.output_dir
            ]
            
            if not use_gpu:
                cmd.append("--no-gpu")
            
            if use_mobile:
                cmd.append("--mobile")
            
            # Exécuter la commande
            logger.info(f"Exécution de la commande: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Erreur lors du traitement: {result.stderr}")
                return ""
            
            # Trouver le fichier de soumission généré
            submission_file = ""
            for file in os.listdir(self.output_dir):
                if file.lower() == "submission.csv":
                    submission_file = os.path.join(self.output_dir, file)
                    break
            
            if not submission_file:
                logger.error("Aucun fichier de soumission généré")
                return ""
            
            logger.info(f"Traitement terminé, fichier de soumission: {submission_file}")
            return submission_file
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement: {str(e)}")
            return ""
    
    def run_complete_workflow(self, use_gpu: bool = True, use_mobile: bool = False) -> bool:
        """
        Exécute le workflow complet: téléchargement, traitement et soumission
        
        Args:
            use_gpu: Utiliser le GPU si disponible
            use_mobile: Utiliser la version mobile du simulateur
            
        Returns:
            True si le workflow a réussi, False sinon
        """
        # Télécharger les données
        if not self.download_competition_data():
            return False
        
        # Organiser les données
        if not self.organize_data_directories():
            return False
        
        # Exécuter le traitement
        submission_file = self.run_neurax_processing(use_gpu=use_gpu, use_mobile=use_mobile)
        if not submission_file:
            return False
        
        # Soumettre les résultats
        message = f"Neurax2 - GPU: {'Oui' if use_gpu else 'Non'}, Mobile: {'Oui' if use_mobile else 'Non'}"
        if not self.submit_results(submission_file, message):
            return False
        
        logger.info("Workflow complet terminé avec succès")
        return True


def main():
    """
    Point d'entrée principal
    """
    parser = argparse.ArgumentParser(description="Intégration Neurax2 avec Kaggle")
    parser.add_argument("--competition", type=str, default="arc-prize-2025", 
                      help="Nom de la compétition Kaggle")
    parser.add_argument("--data-dir", type=str, default="./kaggle_data", 
                      help="Répertoire pour les données téléchargées")
    parser.add_argument("--output-dir", type=str, default="./kaggle_output", 
                      help="Répertoire pour les fichiers de soumission")
    parser.add_argument("--username", type=str, default=None, 
                      help="Nom d'utilisateur Kaggle (facultatif)")
    parser.add_argument("--api-key", type=str, default=None, 
                      help="Clé API Kaggle (facultatif)")
    parser.add_argument("--no-gpu", action="store_true", 
                      help="Désactiver l'utilisation du GPU")
    parser.add_argument("--mobile", action="store_true", 
                      help="Utiliser la version mobile du simulateur")
    parser.add_argument("--download-only", action="store_true", 
                      help="Télécharger uniquement les données sans traitement")
    parser.add_argument("--process-only", action="store_true", 
                      help="Exécuter uniquement le traitement sans téléchargement ni soumission")
    parser.add_argument("--submit-only", type=str, default="", 
                      help="Soumettre uniquement le fichier spécifié")
    
    args = parser.parse_args()
    
    # Créer l'intégration
    integration = KaggleIntegration(
        competition_name=args.competition,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        username=args.username,
        api_key=args.api_key
    )
    
    # Exécuter l'action demandée
    if args.download_only:
        integration.download_competition_data()
        integration.organize_data_directories()
    elif args.process_only:
        integration.run_neurax_processing(use_gpu=not args.no_gpu, use_mobile=args.mobile)
    elif args.submit_only:
        integration.submit_results(args.submit_only)
    else:
        integration.run_complete_workflow(use_gpu=not args.no_gpu, use_mobile=args.mobile)


if __name__ == "__main__":
    main()