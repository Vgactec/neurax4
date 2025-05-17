
import os
import sys
import logging
import subprocess
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_git():
    """Configure Git avec les informations appropriées"""
    try:
        # Supprimer toute configuration Git existante
        if os.path.exists(".git"):
            subprocess.run(["rm", "-rf", ".git"], check=True)
            
        # Initialiser un nouveau dépôt
        subprocess.run(["git", "init"], check=True)
        
        # Configurer les informations utilisateur
        subprocess.run(["git", "config", "--local", "user.name", "Vgactec"], check=True)
        subprocess.run(["git", "config", "--local", "user.email", "vgactec@outlook.fr"], check=True)
        
        logger.info("Configuration Git réussie")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de la configuration Git: {e}")
        return False

def push_changes():
    """Pousse les changements vers GitHub"""
    try:
        if not setup_git():
            return False
            
        # Ajouter la remote de manière sécurisée
        subprocess.run(["git", "remote", "add", "origin", "git@github.com:Vgactec/neurax4.git"], check=True)
        
        # Ajouter tous les fichiers
        subprocess.run(["git", "add", "."], check=True)
        
        # Commit
        subprocess.run(["git", "commit", "-m", "Premier push de tous les fichiers"], check=True)
        
        # Configurer la branche main
        subprocess.run(["git", "branch", "-M", "main"], check=True)
        
        # Push
        subprocess.run(["git", "push", "-u", "origin", "main"], check=True)
        
        logger.info("Push réussi")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Erreur lors du push: {e}")
        return False

if __name__ == "__main__":
    push_changes()
