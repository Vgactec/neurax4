#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script d'analyse du dépôt GitHub "neurax3"
Ce script permet de cloner et d'analyser en détail le contenu
du dépôt pour générer un rapport complet sur sa structure et ses fonctionnalités.
"""

import os
import subprocess
import json
import re
import base64
import requests
from collections import defaultdict, Counter
import markdown
import time
from pathlib import Path

# Configuration
REPO_OWNER = "Vgactec"
REPO_NAME = "neurax3"
REPO_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}.git"
CLONE_DIR = f"./{REPO_NAME}"
API_BASE_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"
RAPPORT_FILE = "rapport_analyse.md"

# Extensions de fichiers à analyser par catégorie
CODE_EXTENSIONS = {
    'python': ['.py'],
    'javascript': ['.js', '.jsx', '.ts', '.tsx'],
    'html': ['.html', '.htm'],
    'css': ['.css', '.scss', '.sass', '.less'],
    'php': ['.php'],
    'java': ['.java'],
    'c/c++': ['.c', '.cpp', '.h', '.hpp'],
    'go': ['.go'],
    'ruby': ['.rb'],
    'shell': ['.sh', '.bash'],
    'autres_code': ['.rs', '.swift', '.kt', '.pl', '.cs']
}

CONFIG_EXTENSIONS = ['.json', '.yml', '.yaml', '.toml', '.ini', '.config', '.xml']
DOC_EXTENSIONS = ['.md', '.txt', '.rst', '.adoc', '.pdf', '.doc', '.docx']
DATA_EXTENSIONS = ['.csv', '.sqlite', '.db', '.json', '.xml', '.yaml', '.yml']

def create_report_header():
    """Crée l'en-tête du rapport d'analyse"""
    report = f"""# Rapport d'Analyse du Dépôt GitHub "{REPO_NAME}"

Date de l'analyse: {time.strftime("%d/%m/%Y %H:%M:%S")}

## Introduction

Ce document présente une analyse détaillée du dépôt GitHub "{REPO_NAME}" appartenant à l'utilisateur "{REPO_OWNER}". 
L'objectif est de comprendre l'architecture, les fonctionnalités et la structure du projet.

"""
    return report

def clone_repository():
    """Clone le dépôt Git pour analyse locale en utilisant l'authentification"""
    print(f"Clonage du dépôt {REPO_URL}...")
    
    # Utilisation d'un token d'authentification GitHub
    token = "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"
    auth_url = f"https://{token}@github.com/{REPO_OWNER}/{REPO_NAME}.git"
    
    if os.path.exists(CLONE_DIR):
        print(f"Le répertoire {CLONE_DIR} existe déjà, suppression...")
        try:
            if os.name == 'nt':  # Windows
                subprocess.run(f"rmdir /s /q {CLONE_DIR}", shell=True, check=True)
            else:  # Unix/Linux/MacOS
                subprocess.run(f"rm -rf {CLONE_DIR}", shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Erreur lors de la suppression du répertoire: {e}")
            # Continuer malgré l'erreur
    
    try:
        # Utiliser l'URL authentifiée pour le clonage
        result = subprocess.run(["git", "clone", auth_url, CLONE_DIR], 
                               capture_output=True, text=True, check=True)
        print("Dépôt cloné avec succès.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors du clonage du dépôt: {e}")
        print(f"Sortie standard: {e.stdout}")
        print(f"Erreur standard: {e.stderr}")
        return False

def get_repo_info():
    """Récupère les informations générales sur le dépôt via l'API GitHub avec authentification"""
    print("Récupération des informations générales du dépôt...")
    
    # Token d'authentification GitHub
    token = "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"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    try:
        response = requests.get(API_BASE_URL, headers=headers)
        response.raise_for_status()
        repo_data = response.json()
        
        # Récupération des branches
        branches_response = requests.get(f"{API_BASE_URL}/branches", headers=headers)
        branches_response.raise_for_status()
        branches = [branch["name"] for branch in branches_response.json()]
        
        # Récupération des collaborateurs
        contributors_response = requests.get(f"{API_BASE_URL}/contributors", headers=headers)
        contributors_response.raise_for_status()
        contributors = [contrib["login"] for contrib in contributors_response.json()]
        
        return {
            "name": repo_data.get("name"),
            "description": repo_data.get("description"),
            "stars": repo_data.get("stargazers_count"),
            "forks": repo_data.get("forks_count"),
            "open_issues": repo_data.get("open_issues_count"),
            "created_at": repo_data.get("created_at"),
            "updated_at": repo_data.get("updated_at"),
            "default_branch": repo_data.get("default_branch"),
            "language": repo_data.get("language"),
            "branches": branches,
            "contributors": contributors,
            "license": repo_data.get("license", {}).get("name") if repo_data.get("license") else "Non spécifiée"
        }
    except requests.RequestException as e:
        print(f"Erreur lors de la récupération des informations du dépôt: {e}")
        return None

def analyze_file_structure():
    """Analyse la structure des fichiers et répertoires du dépôt"""
    print("Analyse de la structure des fichiers...")
    
    if not os.path.exists(CLONE_DIR):
        print(f"Le répertoire {CLONE_DIR} n'existe pas. Le dépôt n'a pas été cloné correctement.")
        return None
    
    file_structure = {
        "directories": [],
        "files": [],
        "file_types": defaultdict(int),
        "file_extensions": Counter(),
        "file_sizes": {},
        "top_level_dirs": [],
        "special_files": []
    }
    
    special_file_patterns = [
        "README", "LICENSE", "Dockerfile", "docker-compose", 
        "requirements.txt", "package.json", "setup.py", 
        "CMakeLists.txt", "Makefile", ".gitignore", ".env"
    ]
    
    total_size = 0
    file_count = 0
    
    for root, dirs, files in os.walk(CLONE_DIR):
        rel_root = os.path.relpath(root, CLONE_DIR)
        if rel_root == ".":
            file_structure["top_level_dirs"] = dirs.copy()
        
        for d in dirs:
            dir_path = os.path.join(rel_root, d)
            if dir_path.startswith("./."):
                dir_path = dir_path[2:]
            elif dir_path == ".":
                dir_path = ""
            file_structure["directories"].append(dir_path)
        
        for f in files:
            file_path = os.path.join(root, f)
            rel_file_path = os.path.join(rel_root, f)
            if rel_file_path.startswith("./."):
                rel_file_path = rel_file_path[2:]
            elif rel_file_path.startswith("./"):
                rel_file_path = rel_file_path[2:]
            
            file_structure["files"].append(rel_file_path)
            
            _, ext = os.path.splitext(f.lower())
            file_structure["file_extensions"][ext] += 1
            
            # Catégorisation des fichiers
            categorized = False
            for category, extensions in CODE_EXTENSIONS.items():
                if ext in extensions:
                    file_structure["file_types"][category] += 1
                    categorized = True
                    break
            
            if not categorized:
                if ext in CONFIG_EXTENSIONS:
                    file_structure["file_types"]["configuration"] += 1
                elif ext in DOC_EXTENSIONS:
                    file_structure["file_types"]["documentation"] += 1
                elif ext in DATA_EXTENSIONS:
                    file_structure["file_types"]["données"] += 1
                else:
                    file_structure["file_types"]["autres"] += 1
            
            # Taille du fichier
            try:
                file_size = os.path.getsize(file_path)
                file_structure["file_sizes"][rel_file_path] = file_size
                total_size += file_size
                file_count += 1
            except OSError:
                print(f"Erreur lors de la récupération de la taille du fichier {file_path}")
            
            # Détection des fichiers spéciaux
            for pattern in special_file_patterns:
                if pattern.lower() in f.lower():
                    file_structure["special_files"].append(rel_file_path)
                    break
    
    file_structure["total_size"] = total_size
    file_structure["file_count"] = file_count
    file_structure["avg_file_size"] = total_size / file_count if file_count > 0 else 0
    
    return file_structure

def analyze_code_files(file_structure):
    """Analyse le contenu des fichiers de code pour identifier les patterns et dépendances"""
    print("Analyse du contenu des fichiers de code...")
    
    code_analysis = {
        "import_patterns": defaultdict(list),
        "function_counts": defaultdict(int),
        "class_counts": defaultdict(int),
        "code_complexity": {},
        "dependency_graph": defaultdict(list),
        "potential_framework_patterns": defaultdict(list)
    }
    
    framework_patterns = {
        "flask": ["from flask import", "app = Flask", "@app.route"],
        "django": ["from django", "urlpatterns", "models.Model"],
        "pytorch": ["import torch", "torch.nn", "torch.optim"],
        "tensorflow": ["import tensorflow", "tf.keras", "tf.data"],
        "react": ["import React", "useState", "useEffect", "render()"],
        "vue": ["import Vue", "new Vue", "createApp"],
        "angular": ["@Component", "@NgModule", "Injectable"],
        "fastapi": ["from fastapi import", "app = FastAPI", "@app.get"],
        "numpy": ["import numpy", "np.array"],
        "pandas": ["import pandas", "pd.DataFrame"],
        "scikit-learn": ["from sklearn", "fit(", "predict("],
        "express": ["express()", "app.use", "app.listen"],
        "laravel": ["use Illuminate", "extends Controller", "Route::get"],
        "spring": ["@Controller", "@Service", "@Repository"],
        "dotnet": ["using System", "namespace", "public class"],
    }
    
    # Patterns d'importation par langage
    import_patterns = {
        ".py": [
            (r"^import\s+([\w.]+)", "import"),
            (r"^from\s+([\w.]+)\s+import", "from import")
        ],
        ".js": [
            (r"import\s+(?:[\w{}]*\s+from\s+)?['\"]([^'\"]+)['\"]", "import"),
            (r"require\(['\"]([^'\"]+)['\"]", "require")
        ],
        ".java": [
            (r"import\s+([\w.]+);", "import")
        ],
        ".go": [
            (r"import\s+[\(\"]([^\"]+)[\"\)]", "import")
        ],
        ".php": [
            (r"use\s+([\w\\]+);", "use"),
            (r"include\(['\"]([^'\"]+)['\"]", "include"),
            (r"require\(['\"]([^'\"]+)['\"]", "require")
        ]
    }
    
    for file_path in file_structure["files"]:
        full_path = os.path.join(CLONE_DIR, file_path)
        _, ext = os.path.splitext(file_path.lower())
        
        # Ignorer les fichiers qui ne sont pas du code
        is_code_file = False
        for category, extensions in CODE_EXTENSIONS.items():
            if ext in extensions:
                is_code_file = True
                break
        
        if not is_code_file:
            continue
            
        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                # Analyse des imports/dependencies
                if ext in import_patterns:
                    for pattern, pattern_type in import_patterns[ext]:
                        matches = re.findall(pattern, content, re.MULTILINE)
                        for match in matches:
                            code_analysis["import_patterns"][ext].append((match, pattern_type, file_path))
                            code_analysis["dependency_graph"][file_path].append(match)
                
                # Analyse des frameworks
                for framework, patterns in framework_patterns.items():
                    for pattern in patterns:
                        if pattern in content:
                            code_analysis["potential_framework_patterns"][framework].append(file_path)
                
                # Comptage des fonctions et classes pour certains langages
                if ext == ".py":
                    # Comptage des fonctions Python (def)
                    functions = re.findall(r"def\s+(\w+)\s*\(", content)
                    code_analysis["function_counts"]["python"] += len(functions)
                    # Comptage des classes Python (class)
                    classes = re.findall(r"class\s+(\w+)(?:\(\w+\))?:", content)
                    code_analysis["class_counts"]["python"] += len(classes)
                
                elif ext in [".js", ".jsx", ".ts", ".tsx"]:
                    # Comptage des fonctions JS/TS (function, =>)
                    functions = re.findall(r"function\s+(\w+)\s*\(", content)
                    functions += re.findall(r"(\w+)\s*=\s*function\s*\(", content)
                    functions += re.findall(r"(\w+)\s*=\s*\([^)]*\)\s*=>", content)
                    code_analysis["function_counts"]["javascript"] += len(functions)
                    # Comptage des classes JS/TS (class)
                    classes = re.findall(r"class\s+(\w+)(?:\s+extends\s+\w+)?", content)
                    code_analysis["class_counts"]["javascript"] += len(classes)
                
                # Calcul simple de complexité (nombre de lignes et de caractères)
                lines = content.split('\n')
                code_analysis["code_complexity"][file_path] = {
                    "lines": len(lines),
                    "characters": len(content)
                }
                
        except Exception as e:
            print(f"Erreur lors de l'analyse du fichier {file_path}: {e}")
    
    return code_analysis

def analyze_readme():
    """Analyse le fichier README du dépôt"""
    print("Analyse du fichier README...")
    
    readme_paths = [
        os.path.join(CLONE_DIR, "README.md"),
        os.path.join(CLONE_DIR, "README"),
        os.path.join(CLONE_DIR, "README.txt"),
        os.path.join(CLONE_DIR, "Readme.md")
    ]
    
    readme_content = None
    
    for path in readme_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    readme_content = f.read()
                break
            except Exception as e:
                print(f"Erreur lors de la lecture du README {path}: {e}")
    
    return readme_content

def analyze_dependencies():
    """Analyse les fichiers de dépendances du projet"""
    print("Analyse des fichiers de dépendances...")
    
    dependency_files = {
        "python": ["requirements.txt", "setup.py", "Pipfile", "pyproject.toml"],
        "javascript": ["package.json", "yarn.lock", "package-lock.json"],
        "php": ["composer.json"],
        "ruby": ["Gemfile"],
        "go": ["go.mod"],
        "java": ["pom.xml", "build.gradle"],
        "docker": ["Dockerfile", "docker-compose.yml", "docker-compose.yaml"]
    }
    
    dependencies = {lang: {} for lang in dependency_files.keys()}
    
    for lang, files in dependency_files.items():
        for file in files:
            file_path = os.path.join(CLONE_DIR, file)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        dependencies[lang][file] = content
                except Exception as e:
                    print(f"Erreur lors de la lecture du fichier de dépendances {file_path}: {e}")
    
    return dependencies

def generate_report(repo_info, file_structure, code_analysis, readme_content, dependencies):
    """Génère un rapport détaillé basé sur les analyses effectuées"""
    print("Génération du rapport d'analyse...")
    
    report = create_report_header()
    
    # Section 1: Informations générales sur le dépôt
    report += """
## 1. Informations Générales sur le Dépôt

"""
    if repo_info:
        report += f"""
- **Nom du dépôt**: {repo_info['name']}
- **Description**: {repo_info['description'] or 'Aucune description fournie'}
- **Langage principal**: {repo_info['language'] or 'Non spécifié'}
- **Étoiles**: {repo_info['stars']}
- **Forks**: {repo_info['forks']}
- **Issues ouvertes**: {repo_info['open_issues']}
- **Créé le**: {repo_info['created_at']}
- **Dernière mise à jour**: {repo_info['updated_at']}
- **Branche par défaut**: {repo_info['default_branch']}
- **Licence**: {repo_info['license']}

### Branches
- {', '.join(repo_info['branches']) if repo_info['branches'] else 'Aucune branche disponible'}

### Contributeurs
- {', '.join(repo_info['contributors']) if repo_info['contributors'] else 'Aucun contributeur listé'}
"""
    else:
        report += "Informations non disponibles. L'API GitHub n'a pas pu être consultée ou a retourné une erreur.\n"
    
    # Section 2: Structure du projet
    report += """
## 2. Structure du Projet

"""
    if file_structure:
        # Répertoires de premier niveau
        report += "### Répertoires de Premier Niveau\n"
        if file_structure["top_level_dirs"]:
            for dir_name in sorted(file_structure["top_level_dirs"]):
                report += f"- {dir_name}/\n"
        else:
            report += "Aucun répertoire de premier niveau trouvé.\n"
        
        # Fichiers spéciaux
        report += "\n### Fichiers Spéciaux et de Configuration\n"
        if file_structure["special_files"]:
            for special_file in sorted(file_structure["special_files"]):
                report += f"- {special_file}\n"
        else:
            report += "Aucun fichier spécial ou de configuration trouvé.\n"
        
        # Statistiques sur les fichiers
        report += "\n### Statistiques des Fichiers\n"
        report += f"- **Nombre total de fichiers**: {file_structure['file_count']}\n"
        report += f"- **Taille totale**: {file_structure['total_size'] / 1024:.2f} KB\n"
        report += f"- **Taille moyenne des fichiers**: {file_structure['avg_file_size'] / 1024:.2f} KB\n"
        
        # Répartition par type de fichier
        report += "\n### Répartition par Type de Fichier\n"
        for file_type, count in sorted(file_structure["file_types"].items(), key=lambda x: x[1], reverse=True):
            report += f"- **{file_type}**: {count} fichier(s)\n"
        
        # Extensions de fichiers les plus courantes
        report += "\n### Extensions de Fichiers les Plus Courantes\n"
        for ext, count in file_structure["file_extensions"].most_common(10):
            if ext:
                report += f"- **{ext}**: {count} fichier(s)\n"
            else:
                report += f"- **Sans extension**: {count} fichier(s)\n"
    else:
        report += "Informations sur la structure des fichiers non disponibles.\n"
    
    # Section 3: Analyse du code
    report += """
## 3. Analyse du Code

"""
    if code_analysis:
        # Frameworks détectés
        report += "### Frameworks et Bibliothèques Détectés\n"
        if code_analysis["potential_framework_patterns"]:
            for framework, files in sorted(code_analysis["potential_framework_patterns"].items()):
                report += f"- **{framework}**: {len(files)} fichier(s)\n"
                # Limiter à 5 exemples de fichiers pour ne pas surcharger le rapport
                for file in sorted(files)[:5]:
                    report += f"  - {file}\n"
                if len(files) > 5:
                    report += f"  - ... et {len(files) - 5} autre(s) fichier(s)\n"
        else:
            report += "Aucun framework ou bibliothèque connu détecté.\n"
        
        # Statistiques de code
        report += "\n### Statistiques de Code\n"
        
        if code_analysis["function_counts"]:
            report += "#### Fonctions par Langage\n"
            for lang, count in sorted(code_analysis["function_counts"].items(), key=lambda x: x[1], reverse=True):
                report += f"- **{lang}**: {count} fonction(s)\n"
        
        if code_analysis["class_counts"]:
            report += "\n#### Classes par Langage\n"
            for lang, count in sorted(code_analysis["class_counts"].items(), key=lambda x: x[1], reverse=True):
                report += f"- **{lang}**: {count} classe(s)\n"
        
        # Fichiers les plus complexes (par nombre de lignes)
        report += "\n### Fichiers les Plus Complexes (par nombre de lignes)\n"
        complex_files = sorted(
            [(file, data["lines"]) for file, data in code_analysis["code_complexity"].items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        for file, lines in complex_files:
            report += f"- **{file}**: {lines} lignes\n"
    else:
        report += "Informations sur l'analyse du code non disponibles.\n"
    
    # Section 4: Dépendances
    report += """
## 4. Dépendances du Projet

"""
    if dependencies:
        has_dependencies = False
        for lang, deps in dependencies.items():
            if deps:
                has_dependencies = True
                report += f"### Dépendances {lang.capitalize()}\n"
                for file, content in deps.items():
                    report += f"\n#### {file}\n"
                    report += "```\n"
                    # Limiter la taille du contenu affiché
                    max_lines = 30
                    content_lines = content.split('\n')
                    if len(content_lines) > max_lines:
                        report += '\n'.join(content_lines[:max_lines])
                        report += f"\n... (tronqué, {len(content_lines) - max_lines} lignes supplémentaires)\n"
                    else:
                        report += content
                    report += "\n```\n"
        
        if not has_dependencies:
            report += "Aucun fichier de dépendances standard n'a été trouvé dans le projet.\n"
    else:
        report += "Informations sur les dépendances non disponibles.\n"
    
    # Section 5: Documentation
    report += """
## 5. Documentation du Projet

"""
    if readme_content:
        report += "### Contenu du README\n"
        report += "```markdown\n"
        # Limiter la taille du README affiché
        max_lines = 50
        readme_lines = readme_content.split('\n')
        if len(readme_lines) > max_lines:
            report += '\n'.join(readme_lines[:max_lines])
            report += f"\n... (tronqué, {len(readme_lines) - max_lines} lignes supplémentaires)\n"
        else:
            report += readme_content
        report += "\n```\n"
    else:
        report += "Aucun fichier README trouvé dans le projet.\n"
    
    # Section 6: Conclusion et recommandations
    report += """
## 6. Conclusion et Recommandations

### Résumé de l'Architecture
"""
    # Déterminer le type d'architecture en fonction de l'analyse
    if file_structure and code_analysis:
        # Tentons de déduire le type d'architecture
        has_frontend = any(framework in code_analysis["potential_framework_patterns"] 
                          for framework in ["react", "vue", "angular"])
        has_backend = any(framework in code_analysis["potential_framework_patterns"] 
                         for framework in ["flask", "django", "express", "fastapi", "laravel", "spring"])
        has_ml = any(framework in code_analysis["potential_framework_patterns"] 
                    for framework in ["pytorch", "tensorflow", "scikit-learn", "numpy", "pandas"])
        
        if has_frontend and has_backend:
            arch_type = "application web full-stack"
        elif has_frontend:
            arch_type = "application frontend"
        elif has_backend:
            arch_type = "application backend/API"
        elif has_ml:
            arch_type = "application de machine learning/data science"
        else:
            arch_type = "application ou bibliothèque"
        
        main_lang = repo_info.get("language", "non déterminé") if repo_info else "non déterminé"
        
        report += f"""
Basé sur l'analyse du code et de la structure du projet, "{REPO_NAME}" semble être une {arch_type} 
principalement développée en {main_lang}. """
        
        if has_ml:
            report += "Ce projet contient des éléments de machine learning ou d'analyse de données. "
        
        # Ajouter des détails sur les frameworks identifiés
        if code_analysis["potential_framework_patterns"]:
            frameworks = list(code_analysis["potential_framework_patterns"].keys())
            report += f"Le projet utilise les frameworks/bibliothèques suivants: {', '.join(frameworks)}. "
        
        # Points forts et faiblesses
        report += """

### Points Forts et Faiblesses

#### Points Forts:
- """
        # Générons quelques points forts probables
        strengths = []
        if readme_content:
            strengths.append("Documentation de base (README) présente")
        
        if any(special_file.endswith('.gitignore') for special_file in file_structure.get("special_files", [])):
            strengths.append("Utilisation de contrôle de version avec .gitignore configuré")
        
        if dependencies and any(deps for deps in dependencies.values()):
            strengths.append("Gestion des dépendances explicite via des fichiers de configuration")
        
        if not strengths:
            strengths.append("Structure de code organisée")
        
        report += '\n- '.join(strengths)
        
        report += """

#### Points à Améliorer:
- """
        # Générons quelques faiblesses probables
        weaknesses = []
        if not readme_content:
            weaknesses.append("Documentation insuffisante (pas de README détaillé)")
        
        if not dependencies or not any(deps for deps in dependencies.values()):
            weaknesses.append("Gestion des dépendances peu claire ou non explicite")
        
        if not weaknesses:
            weaknesses.append("Documentation technique approfondie pourrait être développée")
            weaknesses.append("Tests automatisés pourraient être renforcés")
        
        report += '\n- '.join(weaknesses)
    else:
        report += "Informations insuffisantes pour déterminer l'architecture du projet.\n"
    
    report += """

### Recommandations

Voici quelques recommandations pour améliorer ce projet:

1. **Documentation** : Enrichir la documentation technique, notamment en ajoutant des commentaires dans le code et en complétant le README
2. **Tests** : Améliorer la couverture des tests automatisés
3. **Structure** : Organiser le code selon les meilleures pratiques du langage principal
4. **Dépendances** : Maintenir à jour les dépendances et expliciter les versions requises
5. **Conteneurisation** : Considérer l'utilisation de Docker pour faciliter le déploiement

## 7. Annexes

### Liste complète des fichiers

Voici la liste des 20 premiers fichiers du projet (triés par chemin):
"""
    # Annexe avec liste de fichiers
    if file_structure and file_structure["files"]:
        sorted_files = sorted(file_structure["files"])
        for file in sorted_files[:min(20, len(sorted_files))]:
            report += f"- {file}\n"
        
        if len(sorted_files) > 20:
            report += f"... et {len(sorted_files) - 20} autres fichiers non affichés.\n"
    else:
        report += "Aucun fichier trouvé dans le projet.\n"
    
    return report

def main():
    """Fonction principale exécutant l'analyse complète du dépôt"""
    print(f"Début de l'analyse du dépôt {REPO_OWNER}/{REPO_NAME}")
    
    # Nettoyage préliminaire pour s'assurer que le répertoire est propre
    if os.path.exists(CLONE_DIR):
        print(f"Nettoyage préliminaire du répertoire existant {CLONE_DIR}...")
        try:
            if os.name == 'nt':  # Windows
                subprocess.run(f"rmdir /s /q {CLONE_DIR}", shell=True, check=True)
            else:  # Unix/Linux/MacOS
                subprocess.run(f"rm -rf {CLONE_DIR}", shell=True, check=True)
            print("Répertoire nettoyé avec succès.")
        except Exception as e:
            print(f"Avertissement lors du nettoyage préliminaire: {e}")
            # Tenter un nettoyage plus agressif si nécessaire
            try:
                subprocess.run(f"rm -rf {CLONE_DIR}", shell=True)
            except:
                pass
    
    # Étape 1: Cloner le dépôt
    if not clone_repository():
        with open(RAPPORT_FILE, 'w', encoding='utf-8') as f:
            f.write("# Rapport d'Analyse - ÉCHEC\n\nLe dépôt n'a pas pu être cloné correctement. Veuillez vérifier l'URL et les permissions.")
        return
    
    # Étape 2: Récupérer des informations générales sur le dépôt
    repo_info = get_repo_info()
    
    # Étape 3: Analyser la structure des fichiers
    file_structure = analyze_file_structure()
    
    # Étape 4: Analyser le contenu des fichiers de code
    code_analysis = analyze_code_files(file_structure) if file_structure else None
    
    # Étape 5: Analyser le README
    readme_content = analyze_readme()
    
    # Étape 6: Analyser les dépendances
    dependencies = analyze_dependencies()
    
    # Étape 7: Générer le rapport
    report = generate_report(repo_info, file_structure, code_analysis, readme_content, dependencies)
    
    # Étape 8: Écrire le rapport dans un fichier
    with open(RAPPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Analyse terminée. Le rapport a été généré dans {RAPPORT_FILE}")

if __name__ == "__main__":
    main()
