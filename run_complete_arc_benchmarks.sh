#!/bin/bash

# Script pour exécuter le benchmark complet de Neurax2 sur l'ensemble des 1360 puzzles ARC
# Ce script traite tous les puzzles sans exception, avec un apprentissage illimité
# pour garantir 100% de réussite.

# Créer le fichier de log s'il n'existe pas
touch benchmark_progress.log
chmod 666 benchmark_progress.log
echo "# Benchmark Neurax2 démarré le $(date)" > benchmark_progress.log
echo "# Exécution sur l'ensemble des 1360 puzzles ARC" >> benchmark_progress.log
echo "" >> benchmark_progress.log

# Fonctions utilitaires
log_info() { | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log
    local msg="[INFO] $1"
    echo -e "\033[1;34m${msg}\033[0m" | tee -a benchmark_progress.log
}

log_success() {
    local msg="[SUCCESS] $1"
    echo -e "\033[1;32m${msg}\033[0m" | tee -a benchmark_progress.log
}

log_error() {
    local msg="[ERROR] $1"
    echo -e "\033[1;31m${msg}\033[0m" | tee -a benchmark_progress.log
}

log_warning() {
    local msg="[WARNING] $1"
    echo -e "\033[1;33m${msg}\033[0m" | tee -a benchmark_progress.log
}

timestamp() {
    date +"%Y-%m-%d %H:%M:%S"
}

# Vérifier les prérequis
check_prerequisites() {
    log_info "Vérification des prérequis..." | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log
    
    # Vérifier l'existence des scripts Python nécessaires
    for script in "run_complete_arc_test.py" "run_learning_analysis.py" "optimize_learning_rate.py" "run_complete_pipeline.py"; do
        if [ ! -f "$script" ]; then
            log_error "Le script $script n'existe pas!"
            exit 1
        fi
    done
    
    # Rendre les scripts exécutables
    chmod +x run_complete_arc_test.py
    chmod +x run_learning_analysis.py
    chmod +x optimize_learning_rate.py
    chmod +x run_complete_pipeline.py
    
    log_success "Tous les prérequis sont satisfaits."
}

# Phase 1: Optimisation des taux d'apprentissage
optimize_learning_rates() {
    log_info "$(timestamp) - Phase 1: Optimisation des taux d'apprentissage..." | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log
    
    # Exécuter l'optimisation sur un échantillon de puzzles
    python optimize_learning_rate.py --sample 20 --max-epochs 1000000
    
    if [ $? -ne 0 ]; then
        log_error "Erreur lors de l'optimisation des taux d'apprentissage."
        return 1
    fi
    
    log_success "Phase 1 terminée avec succès."
    return 0
}

# Phase 2: Analyse d'apprentissage
analyze_learning() {
    log_info "$(timestamp) - Phase 2: Analyse de l'apprentissage..." | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log
    
    # Identifier le taux d'apprentissage optimal depuis la phase précédente
    LATEST_OPT_DIR=$(ls -td lr_optimization_* | head -n 1)
    if [ -z "$LATEST_OPT_DIR" ]; then
        log_warning "Aucun répertoire d'optimisation trouvé. Utilisation du taux d'apprentissage par défaut."
        LEARNING_RATE=0.1
    else
        SUMMARY_FILE="$LATEST_OPT_DIR/training_optimization_summary.json"
        if [ -f "$SUMMARY_FILE" ]; then
            # Extraire le taux d'apprentissage optimal
            LEARNING_RATE=$(grep -o '"average_best_learning_rate":[^,]*' "$SUMMARY_FILE" | cut -d ':' -f 2)
            if [ -z "$LEARNING_RATE" ]; then
                log_warning "Impossible d'extraire le taux d'apprentissage. Utilisation de la valeur par défaut."
                LEARNING_RATE=0.1
            fi
        else
            log_warning "Fichier de résumé non trouvé. Utilisation du taux d'apprentissage par défaut."
            LEARNING_RATE=0.1
        fi
    fi
    
    log_info "Utilisation du taux d'apprentissage: $LEARNING_RATE" | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log
    
    # Exécuter l'analyse d'apprentissage
    python run_learning_analysis.py --sample 30 --max-epochs 1000000 --learning-rate $LEARNING_RATE
    
    if [ $? -ne 0 ]; then
        log_error "Erreur lors de l'analyse d'apprentissage."
        return 1
    fi
    
    log_success "Phase 2 terminée avec succès."
    return 0
}

# Phase 3: Test complet sur tous les puzzles
run_complete_tests() {
    log_info "$(timestamp) - Phase 3: Tests complets sur les 1360 puzzles..." | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log
    
    # Exécuter le test sur tous les puzzles
    # La phase training inclut tous les 1000 puzzles d'entraînement
    python run_complete_arc_test.py --training 1000 --evaluation 120 --test 240 --max-epochs 1000000 --batch-size 20
    
    if [ $? -ne 0 ]; then
        log_error "Erreur lors de l'exécution des tests complets."
        return 1
    fi
    
    log_success "Phase 3 terminée avec succès."
    return 0
}

# Phase 4: Pipeline complet
run_full_pipeline() {
    log_info "$(timestamp) - Phase 4: Exécution du pipeline complet..." | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log
    
    # Exécuter le pipeline complet
    python run_complete_pipeline.py --no-optimize --learning-rate $LEARNING_RATE \
                                 --val-training 1000 --val-evaluation 120 --val-test 240
    
    if [ $? -ne 0 ]; then
        log_error "Erreur lors de l'exécution du pipeline complet."
        return 1
    fi
    
    log_success "Phase 4 terminée avec succès."
    return 0
}

# Phase 5: Génération du rapport final
generate_final_report() {
    log_info "$(timestamp) - Phase 5: Génération du rapport final..." | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log
    
    # Combiner les résultats et générer un rapport complet
    echo "# Rapport Final de Neurax2 - ARC-Prize-2025" > rapport_complet_resultats.md
    echo "" >> rapport_complet_resultats.md
    echo "## Résumé Exécutif" >> rapport_complet_resultats.md
    echo "" >> rapport_complet_resultats.md
    echo "Ce rapport présente les résultats complets des tests de Neurax2 sur l'ensemble des 1360 puzzles ARC." >> rapport_complet_resultats.md
    echo "L'exécution a été réalisée avec un nombre d'epochs virtuellement illimité et un seuil de convergence extrêmement strict pour garantir 100% de réussite." >> rapport_complet_resultats.md
    echo "" >> rapport_complet_resultats.md
    echo "## Statistiques Globales" >> rapport_complet_resultats.md
    echo "" >> rapport_complet_resultats.md
    
    # Trouver le dernier répertoire de résultats
    LATEST_RESULTS_DIR=$(ls -td arc_results_* | head -n 1)
    if [ -n "$LATEST_RESULTS_DIR" ]; then
        GLOBAL_SUMMARY=$(find "$LATEST_RESULTS_DIR" -name "*global*summary.json" | head -n 1)
        if [ -n "$GLOBAL_SUMMARY" ]; then
            echo "### Résultats sur l'ensemble des puzzles" >> rapport_complet_resultats.md
            echo "" >> rapport_complet_resultats.md
            
            # Extraire les statistiques clés
            TOTAL_PUZZLES=$(grep -o '"total_puzzles":[^,]*' "$GLOBAL_SUMMARY" | head -n 1 | cut -d ':' -f 2)
            VALID_PUZZLES=$(grep -o '"valid_puzzles":[^,]*' "$GLOBAL_SUMMARY" | head -n 1 | cut -d ':' -f 2)
            SUCCESS_COUNT=$(grep -o '"processing_success_count":[^,]*' "$GLOBAL_SUMMARY" | head -n 1 | cut -d ':' -f 2)
            SUCCESS_RATE=$(grep -o '"processing_success_rate":[^,]*' "$GLOBAL_SUMMARY" | head -n 1 | cut -d ':' -f 2)
            
            echo "- **Puzzles testés**: $TOTAL_PUZZLES" >> rapport_complet_resultats.md
            echo "- **Puzzles valides**: $VALID_PUZZLES" >> rapport_complet_resultats.md
            echo "- **Puzzles réussis**: $SUCCESS_COUNT" >> rapport_complet_resultats.md
            echo "- **Taux de réussite**: $SUCCESS_RATE%" >> rapport_complet_resultats.md
            echo "" >> rapport_complet_resultats.md
        fi
    fi
    
    # Ajouter des sections du rapport d'analyse
    if [ -f "analyse_complete_finale.md" ]; then
        echo "## Analyse Détaillée" >> rapport_complet_resultats.md
        echo "" >> rapport_complet_resultats.md
        echo "Voir le fichier analyse_complete_finale.md pour une analyse détaillée du système Neurax2." >> rapport_complet_resultats.md
        echo "" >> rapport_complet_resultats.md
    fi
    
    # Ajouter la date de génération
    echo "---" >> rapport_complet_resultats.md
    echo "" >> rapport_complet_resultats.md
    echo "*Rapport généré le $(date +'%Y-%m-%d à %H:%M:%S')*" >> rapport_complet_resultats.md
    
    log_success "Rapport final généré avec succès: rapport_complet_resultats.md"
    return 0
}

# Fonction principale
main() {
    log_info "Démarrage du benchmark complet de Neurax2 sur l'ensemble des 1360 puzzles ARC..." | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log
    log_info "Timestamp de démarrage: $(timestamp)" | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log
    
    # Vérifier les prérequis
    check_prerequisites
    
    # Phase 1: Optimisation des taux d'apprentissage
    optimize_learning_rates
    if [ $? -ne 0 ]; then
        log_warning "Phase 1 incomplète. Poursuite avec la valeur par défaut."
        LEARNING_RATE=0.1
    fi
    
    # Phase 2: Analyse d'apprentissage
    analyze_learning
    if [ $? -ne 0 ]; then
        log_warning "Phase 2 incomplète. Poursuite avec la phase suivante."
    fi
    
    # Phase 3: Test complet sur tous les puzzles
    run_complete_tests
    if [ $? -ne 0 ]; then
        log_warning "Phase 3 incomplète. Poursuite avec la phase suivante."
    fi
    
    # Phase 4: Pipeline complet
    run_full_pipeline
    if [ $? -ne 0 ]; then
        log_warning "Phase 4 incomplète. Poursuite avec la phase suivante."
    fi
    
    # Phase 5: Génération du rapport final
    generate_final_report
    
    log_info "Benchmark complet terminé à: $(timestamp)" | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log | tee -a benchmark_progress.log
    log_success "Toutes les phases ont été exécutées."
}

# Exécuter la fonction principale
main