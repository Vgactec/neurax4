#!/usr/bin/env python3
"""
Script pour mettre à jour le notebook existant sur Kaggle et récupérer les logs d'erreur
"""

import os
import json
import subprocess
import time
import sys
from pathlib import Path

def check_kaggle_credentials():
    """Vérifie que les identifiants Kaggle sont correctement configurés"""
    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        print("ERREUR: Identifiants Kaggle manquants")
        print("Veuillez définir KAGGLE_USERNAME et KAGGLE_KEY")
        return False
    return True

def update_existing_notebook(notebook_path, target_notebook):
    """Met à jour un notebook existant sur Kaggle"""
    try:
        print(f"Mise à jour du notebook {target_notebook}...")
        
        # Préparer le fichier metadata
        metadata_path = "kernel-metadata.json"
        metadata = {
            "id": target_notebook,
            "title": "Neurax3 ARC System - Optimisé pour traitement complet",
            "code_file": notebook_path,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": ["kaggle/arc-prize-2025"],
            "competition_sources": ["arc-prize-2025"],
            "kernel_sources": []
        }
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Pousser le notebook
        cmd = ["kaggle", "kernels", "push", "-p", "."]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Mise à jour réussie!")
            kernel_id = metadata["id"]
            print(f"Notebook disponible à: https://www.kaggle.com/code/{kernel_id}")
            return kernel_id
        else:
            print("ERREUR: Échec de la mise à jour du notebook")
            print(f"Erreur: {result.stderr}")
            return None
    except Exception as e:
        print(f"ERREUR: Exception lors de la mise à jour: {str(e)}")
        return None

def fetch_notebook_logs(kernel_id):
    """Récupère les logs d'exécution du notebook"""
    try:
        print(f"Récupération des logs pour {kernel_id}...")
        
        # Attente que le notebook commence à s'exécuter
        print("Attente du démarrage de l'exécution...")
        time.sleep(10)
        
        # Récupération des logs
        cmd = ["kaggle", "kernels", "output", kernel_id]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Logs récupérés avec succès!")
            log_content = result.stdout
            
            # Enregistrement des logs dans un fichier
            log_path = "kaggle_execution_logs.txt"
            with open(log_path, "w") as f:
                f.write(log_content)
            
            print(f"Logs enregistrés dans {log_path}")
            
            # Afficher les dernières lignes des logs
            if log_content:
                print("\nDernières lignes des logs:")
                lines = log_content.splitlines()
                for line in lines[-20:]:
                    print(line)
            else:
                print("Aucun log disponible pour le moment.")
            
            return log_path
        else:
            print("ERREUR: Échec de la récupération des logs")
            print(f"Erreur: {result.stderr}")
            return None
    except Exception as e:
        print(f"ERREUR: Exception lors de la récupération des logs: {str(e)}")
        return None

def main():
    """Fonction principale"""
    if not check_kaggle_credentials():
        sys.exit(1)
    
    # Trouver le notebook principal
    notebook_path = "kaggle-kernel/kernel.ipynb"
    if not Path(notebook_path).exists():
        print(f"ERREUR: Notebook {notebook_path} introuvable")
        sys.exit(1)
    
    # Copier le notebook et le metadata dans le répertoire courant
    subprocess.run(["cp", notebook_path, "."], check=True)
    
    # Mettre à jour le notebook existant
    kernel_id = update_existing_notebook("kernel.ipynb", "ndarray2000/neurax3-arc-system-for-arc-prize-2025")
    
    if kernel_id:
        # Récupérer les logs d'exécution
        fetch_notebook_logs(kernel_id)
    
    # Nettoyer les fichiers temporaires
    subprocess.run(["rm", "kernel.ipynb", "kernel-metadata.json"], check=True)

if __name__ == "__main__":
    main()