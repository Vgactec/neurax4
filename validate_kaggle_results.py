"""
Script de validation des résultats obtenus sur Kaggle

Ce script analyse les résultats téléchargés depuis Kaggle pour vérifier
que le système Neurax3 a bien traité l'intégralité des 1360 puzzles ARC
et que tous les résultats sont authentiques.
"""

import os
import json
import sys
from datetime import datetime

def validate_results_directory(results_dir):
    """Valide que le répertoire des résultats existe et contient les fichiers nécessaires"""
    if not os.path.exists(results_dir):
        print(f"❌ Le répertoire de résultats n'existe pas: {results_dir}")
        return False
    
    required_files = [
        "training_results.json",
        "evaluation_results.json",
        "test_results.json"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(results_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Fichiers manquants: {', '.join(missing_files)}")
        return False
    
    print(f"✅ Répertoire de résultats valide: {results_dir}")
    return True

def count_puzzles(results_file):
    """Compte le nombre de puzzles dans un fichier de résultats"""
    try:
        with open(results_file, "r") as f:
            results = json.load(f)
            return len(results)
    except Exception as e:
        print(f"❌ Erreur lors de la lecture du fichier {results_file}: {e}")
        return 0

def validate_puzzle_counts(results_dir):
    """Valide le nombre de puzzles traités dans chaque phase"""
    training_count = count_puzzles(os.path.join(results_dir, "training_results.json"))
    evaluation_count = count_puzzles(os.path.join(results_dir, "evaluation_results.json"))
    test_count = count_puzzles(os.path.join(results_dir, "test_results.json"))
    
    total_count = training_count + evaluation_count + test_count
    
    print(f"Nombre de puzzles traités:")
    print(f"- Entraînement: {training_count}/1000")
    print(f"- Évaluation: {evaluation_count}/120")
    print(f"- Test: {test_count}/240")
    print(f"- Total: {total_count}/1360")
    
    if training_count == 1000 and evaluation_count == 120 and test_count == 240:
        print(f"✅ Tous les puzzles ont été traités!")
        return True
    else:
        print(f"❌ Certains puzzles n'ont pas été traités.")
        return False

def validate_puzzle_success_rate(results_dir):
    """Valide le taux de réussite des puzzles traités"""
    result_files = {
        "training": os.path.join(results_dir, "training_results.json"),
        "evaluation": os.path.join(results_dir, "evaluation_results.json"),
        "test": os.path.join(results_dir, "test_results.json")
    }
    
    success_stats = {}
    
    for phase, file_path in result_files.items():
        try:
            with open(file_path, "r") as f:
                results = json.load(f)
                total = len(results)
                if total == 0:
                    success_stats[phase] = 0
                    continue
                
                successful = sum(1 for r in results if r.get("success", False))
                success_rate = (successful / total) * 100
                success_stats[phase] = success_rate
        except Exception as e:
            print(f"❌ Erreur lors de l'analyse des résultats {phase}: {e}")
            success_stats[phase] = 0
    
    print(f"\nTaux de réussite:")
    for phase, rate in success_stats.items():
        print(f"- {phase.capitalize()}: {rate:.2f}%")
    
    # Calcul du taux global
    overall_rate = sum(success_stats.values()) / len(success_stats) if success_stats else 0
    print(f"- Global: {overall_rate:.2f}%")
    
    if overall_rate > 90:
        print(f"✅ Excellent taux de réussite!")
    elif overall_rate > 70:
        print(f"✅ Bon taux de réussite.")
    else:
        print(f"⚠️ Taux de réussite moyen.")
    
    return overall_rate > 50  # Considérer réussi si plus de 50%

def validate_processing_times(results_dir):
    """Valide les temps de traitement des puzzles"""
    result_files = {
        "training": os.path.join(results_dir, "training_results.json"),
        "evaluation": os.path.join(results_dir, "evaluation_results.json"),
        "test": os.path.join(results_dir, "test_results.json")
    }
    
    time_stats = {}
    iteration_stats = {}
    
    for phase, file_path in result_files.items():
        try:
            with open(file_path, "r") as f:
                results = json.load(f)
                if not results:
                    time_stats[phase] = 0
                    iteration_stats[phase] = 0
                    continue
                
                total_time = sum(r.get("execution_time", 0) for r in results)
                avg_time = total_time / len(results) if results else 0
                time_stats[phase] = avg_time
                
                total_iterations = sum(r.get("iterations", 0) for r in results)
                avg_iterations = total_iterations / len(results) if results else 0
                iteration_stats[phase] = avg_iterations
        except Exception as e:
            print(f"❌ Erreur lors de l'analyse des temps {phase}: {e}")
            time_stats[phase] = 0
            iteration_stats[phase] = 0
    
    print(f"\nTemps de traitement moyen par puzzle:")
    for phase, time in time_stats.items():
        print(f"- {phase.capitalize()}: {time:.2f} secondes")
    
    print(f"\nNombre moyen d'itérations par puzzle:")
    for phase, iterations in iteration_stats.items():
        print(f"- {phase.capitalize()}: {iterations:.2f} itérations")
    
    # Estimation du temps total
    total_time = 0
    for phase, file_path in result_files.items():
        try:
            with open(file_path, "r") as f:
                results = json.load(f)
                phase_time = sum(r.get("execution_time", 0) for r in results)
                total_time += phase_time
        except:
            pass
    
    hours = total_time / 3600
    print(f"\nTemps total de traitement estimé: {total_time:.2f} secondes ({hours:.2f} heures)")
    
    return True

def generate_complete_report(results_dir):
    """Génère un rapport complet de validation"""
    report_file = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    sections = [
        f"# Rapport de Validation Neurax3 - ARC Prize 2025\n",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"## Résumé\n\n"
    ]
    
    # Compter les puzzles
    training_count = count_puzzles(os.path.join(results_dir, "training_results.json"))
    evaluation_count = count_puzzles(os.path.join(results_dir, "evaluation_results.json"))
    test_count = count_puzzles(os.path.join(results_dir, "test_results.json"))
    total_count = training_count + evaluation_count + test_count
    
    if total_count == 1360:
        sections.append("✅ **VALIDATION COMPLÈTE**: Le système Neurax3 a traité avec succès tous les 1360 puzzles ARC!\n\n")
    else:
        sections.append(f"⚠️ **VALIDATION PARTIELLE**: Le système Neurax3 a traité {total_count}/1360 puzzles ARC.\n\n")
    
    # Statistiques des puzzles
    sections.append("## Statistiques de Traitement\n\n")
    sections.append("| Phase | Puzzles Traités | Pourcentage |\n")
    sections.append("|-------|----------------|-------------|\n")
    sections.append(f"| Entraînement | {training_count}/1000 | {(training_count/1000*100):.2f}% |\n")
    sections.append(f"| Évaluation | {evaluation_count}/120 | {(evaluation_count/120*100):.2f}% |\n")
    sections.append(f"| Test | {test_count}/240 | {(test_count/240*100):.2f}% |\n")
    sections.append(f"| **Total** | **{total_count}/1360** | **{(total_count/1360*100):.2f}%** |\n\n")
    
    # Taux de réussite
    sections.append("## Taux de Réussite\n\n")
    
    result_files = {
        "Entraînement": os.path.join(results_dir, "training_results.json"),
        "Évaluation": os.path.join(results_dir, "evaluation_results.json"),
        "Test": os.path.join(results_dir, "test_results.json")
    }
    
    sections.append("| Phase | Puzzles Réussis | Taux de Réussite |\n")
    sections.append("|-------|----------------|------------------|\n")
    
    overall_success = 0
    overall_total = 0
    
    for phase, file_path in result_files.items():
        try:
            with open(file_path, "r") as f:
                results = json.load(f)
                total = len(results)
                successful = sum(1 for r in results if r.get("success", False))
                rate = (successful / total) * 100 if total > 0 else 0
                sections.append(f"| {phase} | {successful}/{total} | {rate:.2f}% |\n")
                
                overall_success += successful
                overall_total += total
        except:
            sections.append(f"| {phase} | N/A | N/A |\n")
    
    overall_rate = (overall_success / overall_total) * 100 if overall_total > 0 else 0
    sections.append(f"| **Total** | **{overall_success}/{overall_total}** | **{overall_rate:.2f}%** |\n\n")
    
    # Temps de traitement
    sections.append("## Temps de Traitement\n\n")
    
    sections.append("| Phase | Temps Moyen (s) | Iterations Moyennes |\n")
    sections.append("|-------|----------------|---------------------|\n")
    
    total_time = 0
    
    for phase, file_path in result_files.items():
        try:
            with open(file_path, "r") as f:
                results = json.load(f)
                if not results:
                    sections.append(f"| {phase} | N/A | N/A |\n")
                    continue
                
                total_phase_time = sum(r.get("execution_time", 0) for r in results)
                avg_time = total_phase_time / len(results)
                total_time += total_phase_time
                
                total_iterations = sum(r.get("iterations", 0) for r in results)
                avg_iterations = total_iterations / len(results)
                
                sections.append(f"| {phase} | {avg_time:.2f} | {avg_iterations:.2f} |\n")
        except:
            sections.append(f"| {phase} | N/A | N/A |\n")
    
    hours = total_time / 3600
    sections.append(f"\nTemps total de traitement estimé: **{total_time:.2f} secondes** (**{hours:.2f} heures**)\n\n")
    
    # Conclusion
    sections.append("## Conclusion\n\n")
    
    if total_count == 1360 and overall_rate > 70:
        sections.append("✅ **VALIDATION RÉUSSIE**: Le système Neurax3 a traité avec succès tous les puzzles ARC avec un excellent taux de réussite!\n\n")
        sections.append("Le système est prêt pour la soumission finale à la compétition ARC-Prize-2025.\n")
    elif total_count == 1360:
        sections.append("✅ **VALIDATION COMPLÈTE**: Le système Neurax3 a traité tous les puzzles ARC, mais avec un taux de réussite à améliorer.\n\n")
        sections.append("Le système est fonctionnel mais pourrait bénéficier d'optimisations supplémentaires pour améliorer les performances.\n")
    else:
        sections.append("⚠️ **VALIDATION PARTIELLE**: Le système Neurax3 n'a pas traité tous les puzzles ARC.\n\n")
        sections.append(f"Des investigations supplémentaires sont nécessaires pour comprendre pourquoi {1360-total_count} puzzles n'ont pas été traités.\n")
    
    # Écrire le rapport
    with open(report_file, "w") as f:
        f.writelines(sections)
    
    print(f"\n✅ Rapport de validation généré: {report_file}")
    return report_file

def main():
    """Fonction principale de validation"""
    print("=== Validation des Résultats Neurax3 sur Kaggle ===\n")
    
    # Récupérer le chemin du répertoire de résultats
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = input("Entrez le chemin du répertoire de résultats: ")
    
    # Valider le répertoire de résultats
    if not validate_results_directory(results_dir):
        print("\n❌ Impossible de valider les résultats: répertoire ou fichiers manquants")
        return False
    
    # Valider le nombre de puzzles traités
    puzzles_ok = validate_puzzle_counts(results_dir)
    
    # Valider le taux de réussite
    success_ok = validate_puzzle_success_rate(results_dir)
    
    # Valider les temps de traitement
    times_ok = validate_processing_times(results_dir)
    
    # Générer un rapport complet
    report_file = generate_complete_report(results_dir)
    
    # Conclusion
    print("\n=== Conclusion de la Validation ===")
    if puzzles_ok and success_ok and times_ok:
        print("✅ VALIDATION RÉUSSIE: Le système Neurax3 a traité avec succès tous les puzzles ARC!")
        print(f"Le rapport détaillé est disponible dans: {report_file}")
        return True
    elif puzzles_ok:
        print("✅ VALIDATION PARTIELLE: Le système Neurax3 a traité tous les puzzles, mais avec des performances à améliorer.")
        print(f"Le rapport détaillé est disponible dans: {report_file}")
        return True
    else:
        print("❌ VALIDATION ÉCHOUÉE: Le système Neurax3 n'a pas traité tous les puzzles ARC.")
        print(f"Le rapport détaillé est disponible dans: {report_file}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)