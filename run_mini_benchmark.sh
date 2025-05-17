#!/bin/bash

# Script pour exécuter un mini-benchmark de Neurax2
# Ce script traite un échantillon des puzzles ARC pour démontrer le fonctionnement du système

# Créer le fichier de log s'il n'existe pas
touch mini_benchmark_progress.log
chmod 666 mini_benchmark_progress.log
echo "# Mini-Benchmark Neurax2 démarré le $(date)" > mini_benchmark_progress.log
echo "# Exécution sur un échantillon des puzzles ARC" >> mini_benchmark_progress.log
echo "" >> mini_benchmark_progress.log

# Fonctions utilitaires
log_info() {
    local msg="[INFO] $1"
    echo -e "\033[1;34m${msg}\033[0m" | tee -a mini_benchmark_progress.log
}

log_success() {
    local msg="[SUCCESS] $1"
    echo -e "\033[1;32m${msg}\033[0m" | tee -a mini_benchmark_progress.log
}

log_error() {
    local msg="[ERROR] $1"
    echo -e "\033[1;31m${msg}\033[0m" | tee -a mini_benchmark_progress.log
}

timestamp() {
    date +"%Y-%m-%d %H:%M:%S"
}

# Phase 1: Optimisation des taux d'apprentissage
optimize_learning_rates() {
    log_info "$(timestamp) - Phase 1: Optimisation des taux d'apprentissage..."
    
    # Exécuter l'optimisation sur un petit échantillon de puzzles
    python optimize_learning_rate.py --sample 5 --max-epochs 100
    
    if [ $? -ne 0 ]; then
        log_error "Erreur lors de l'optimisation des taux d'apprentissage."
        return 1
    fi
    
    log_success "Phase 1 terminée avec succès."
    return 0
}

# Phase 2: Analyse d'apprentissage
analyze_learning() {
    log_info "$(timestamp) - Phase 2: Analyse de l'apprentissage..."
    
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
    
    log_info "Utilisation du taux d'apprentissage: $LEARNING_RATE"
    
    # Exécuter l'analyse d'apprentissage
    python run_learning_analysis.py --sample 5 --max-epochs 100 --learning-rate $LEARNING_RATE
    
    if [ $? -ne 0 ]; then
        log_error "Erreur lors de l'analyse d'apprentissage."
        return 1
    fi
    
    log_success "Phase 2 terminée avec succès."
    return 0
}

# Phase 3: Test complet sur un échantillon des puzzles
run_sample_tests() {
    log_info "$(timestamp) - Phase 3: Tests sur un échantillon des puzzles..."
    
    # Exécuter le test sur un échantillon des puzzles
    python run_complete_arc_test.py --training 5 --evaluation 5 --test 5 --max-epochs 50 --batch-size 2
    
    if [ $? -ne 0 ]; then
        log_error "Erreur lors de l'exécution des tests."
        return 1
    fi
    
    log_success "Phase 3 terminée avec succès."
    return 0
}

# Fonction principale
main() {
    log_info "Démarrage du mini-benchmark de Neurax2 sur un échantillon des puzzles ARC..."
    log_info "Timestamp de démarrage: $(timestamp)"
    
    # Phase 1: Optimisation des taux d'apprentissage
    optimize_learning_rates
    if [ $? -ne 0 ]; then
        log_error "Phase 1 incomplète."
        LEARNING_RATE=0.1
    fi
    
    # Phase 2: Analyse d'apprentissage
    analyze_learning
    if [ $? -ne 0 ]; then
        log_error "Phase 2 incomplète."
    fi
    
    # Phase 3: Test complet sur un échantillon des puzzles
    run_sample_tests
    if [ $? -ne 0 ]; then
        log_error "Phase 3 incomplète."
    fi
    
    log_info "Mini-benchmark terminé à: $(timestamp)"
    log_success "Toutes les phases ont été exécutées."
}

# Exécuter la fonction principale
main