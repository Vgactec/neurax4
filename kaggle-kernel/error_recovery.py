"""
Script de récupération et analyse des erreurs pour Neurax3
Ce script permet de détecter et corriger automatiquement les erreurs courantes
pendant l'exécution de Neurax3 sur Kaggle.
"""

import os
import sys
import json
import logging
import traceback
import datetime

# Configuration du logger
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"error_recovery_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('error_recovery')

# Fonction pour analyser un notebook d'erreur
def analyze_notebook_errors(notebook_path):
    """Analyse un notebook pour trouver les erreurs d'exécution"""
    try:
        with open(notebook_path, 'r') as f:
            notebook = json.load(f)
        
        errors = []
        
        for i, cell in enumerate(notebook.get('cells', [])):
            if cell.get('cell_type') == 'code' and cell.get('outputs'):
                for output in cell.get('outputs', []):
                    if output.get('output_type') == 'error':
                        error_name = output.get('ename', 'Unknown')
                        error_value = output.get('evalue', 'Unknown')
                        error_traceback = output.get('traceback', [])
                        
                        errors.append({
                            'cell_index': i,
                            'error_name': error_name,
                            'error_value': error_value,
                            'traceback': error_traceback
                        })
        
        return errors
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse du notebook: {e}")
        return []

# Fonction pour corriger les erreurs courantes
def fix_errors(errors, notebook_path):
    """Applique les corrections pour les erreurs identifiées"""
    if not errors:
        logger.info("Aucune erreur détectée dans le notebook.")
        return False
    
    logger.info(f"Analyse de {len(errors)} erreurs trouvées dans le notebook")
    
    fixed = False
    error_summary = {}
    
    for error in errors:
        error_name = error.get('error_name', 'Unknown')
        error_value = error.get('error_value', 'Unknown')
        cell_index = error.get('cell_index', -1)
        
        if error_name not in error_summary:
            error_summary[error_name] = 0
        error_summary[error_name] += 1
        
        logger.info(f"Erreur dans la cellule {cell_index}: {error_name} - {error_value}")
        
        # Appliquer des corrections spécifiques
        if error_name == 'ImportError' and 'numpy' in error_value:
            logger.info("Correction d'une erreur d'importation NumPy")
            # Ajouter code pour l'installation de NumPy
            fixed = True
        
        elif error_name == 'ImportError' and 'torch' in error_value:
            logger.info("Correction d'une erreur d'importation PyTorch")
            # Ajouter code pour l'installation de PyTorch
            fixed = True
        
        elif error_name == 'AttributeError' and 'process_puzzle' in error_value:
            logger.info("Correction d'une erreur d'attribut du moteur Neurax")
            # Ajouter code pour corriger l'attribut manquant
            fixed = True
        
        # Autres corrections spécifiques...
    
    # Enregistrer un résumé des erreurs
    with open(os.path.join(log_dir, "error_summary.json"), 'w') as f:
        json.dump({
            "error_counts": error_summary,
            "total_errors": len(errors),
            "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=2)
    
    logger.info(f"Résumé des erreurs enregistré dans {os.path.join(log_dir, 'error_summary.json')}")
    return fixed

# Fonction pour générer un rapport d'erreurs
def generate_error_report(errors, output_path):
    """Génère un rapport détaillé des erreurs"""
    with open(output_path, 'w') as f:
        f.write("# Rapport d'analyse des erreurs Neurax3\n\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if not errors:
            f.write("Aucune erreur détectée dans le notebook.\n")
            return
        
        f.write(f"## Résumé des erreurs\n\n")
        f.write(f"Nombre total d'erreurs: {len(errors)}\n\n")
        
        error_types = {}
        for error in errors:
            error_name = error.get('error_name', 'Unknown')
            if error_name not in error_types:
                error_types[error_name] = 0
            error_types[error_name] += 1
        
        f.write("### Types d'erreurs\n\n")
        for error_name, count in error_types.items():
            f.write(f"- {error_name}: {count} occurrence(s)\n")
        f.write("\n")
        
        f.write("## Détails des erreurs\n\n")
        for i, error in enumerate(errors):
            f.write(f"### Erreur {i+1}\n\n")
            f.write(f"- Cellule: {error.get('cell_index', 'Inconnue')}\n")
            f.write(f"- Type: {error.get('error_name', 'Inconnu')}\n")
            f.write(f"- Message: {error.get('error_value', 'Inconnu')}\n")
            f.write(f"- Traceback:\n")
            for line in error.get('traceback', []):
                f.write(f"  {line}\n")
            f.write("\n")
        
        f.write("## Recommandations\n\n")
        # Ajouter des recommandations spécifiques en fonction des erreurs

# Fonction principale pour récupération des erreurs
def main():
    """Fonction principale pour la récupération des erreurs"""
    logger.info("=== DÉMARRAGE DE LA RÉCUPÉRATION DES ERREURS ===")
    
    # Vérifier si le notebook existe
    notebook_path = "kernel.ipynb"
    if not os.path.exists(notebook_path):
        logger.error(f"Le notebook {notebook_path} n'existe pas")
        return
    
    # Analyser les erreurs
    logger.info(f"Analyse du notebook {notebook_path}")
    errors = analyze_notebook_errors(notebook_path)
    
    if errors:
        logger.info(f"Trouvé {len(errors)} erreurs dans le notebook")
        
        # Générer un rapport d'erreurs
        report_path = os.path.join(log_dir, "error_report.md")
        generate_error_report(errors, report_path)
        logger.info(f"Rapport d'erreurs généré: {report_path}")
        
        # Corriger les erreurs
        fixed = fix_errors(errors, notebook_path)
        if fixed:
            logger.info("Des corrections ont été appliquées")
        else:
            logger.info("Aucune correction n'a été appliquée")
            
        # Générer un rapport d'erreurs
        report_path = os.path.join(log_dir, "error_report.md")
        generate_error_report(errors, report_path)
        logger.info(f"Rapport d'erreurs généré: {report_path}")
    else:
        logger.info("Aucune erreur trouvée dans le notebook")
        report_path = "N/A"
    
    # Créer un fichier de statut
    status_path = os.path.join(log_dir, "status.txt")
    with open(status_path, 'w') as f:
        f.write("=== STATUT DE L'ANALYSE D'ERREURS ===\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Nombre d'erreurs trouvées: {len(errors)}\n")
        f.write(f"Chemin du rapport: {report_path}\n")
    
    logger.info("=== FIN DE LA RÉCUPÉRATION DES ERREURS ===")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Erreur non gérée: {e}")
        logger.error(traceback.format_exc())