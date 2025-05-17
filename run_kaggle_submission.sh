#!/bin/bash

# Script pour finaliser le projet Neurax2 et soumettre les résultats à Kaggle

# Fonctions utilitaires
log_info() {
    local msg="[INFO] $1"
    echo -e "\033[1;34m${msg}\033[0m"
}

log_success() {
    local msg="[SUCCESS] $1"
    echo -e "\033[1;32m${msg}\033[0m"
}

log_error() {
    local msg="[ERROR] $1"
    echo -e "\033[1;31m${msg}\033[0m"
}

timestamp() {
    date +"%Y-%m-%d %H:%M:%S"
}

# Phase 1: Vérification du système
verify_system() {
    log_info "$(timestamp) - Phase 1: Vérification du système Neurax2..."
    
    python verify_complete_system.py --fix --report
    
    if [ $? -ne 0 ]; then
        log_error "Échec de la vérification du système"
        return 1
    fi
    
    log_success "Système vérifié avec succès"
    return 0
}

# Phase 2: Génération des fichiers finaux
generate_final_files() {
    log_info "$(timestamp) - Phase 2: Génération des fichiers finaux..."
    
    # Générer le mini-rapport
    python generate_mini_report.py
    
    if [ $? -ne 0 ]; then
        log_error "Échec de la génération du mini-rapport"
        return 1
    fi
    
    log_success "Fichiers finaux générés avec succès"
    return 0
}

# Phase 3: Préparation pour Kaggle
prepare_kaggle() {
    log_info "$(timestamp) - Phase 3: Préparation des données Kaggle..."
    
    # Télécharger et organiser les données Kaggle
    python kaggle_neurax_integration.py --download-only
    
    if [ $? -ne 0 ]; then
        log_error "Échec de la préparation des données Kaggle"
        return 1
    fi
    
    log_success "Données Kaggle préparées avec succès"
    return 0
}

# Phase 4: Exécution des tests sur Kaggle
run_kaggle_tests() {
    log_info "$(timestamp) - Phase 4: Exécution des tests sur les données Kaggle..."
    
    # Exécuter les tests uniquement
    python kaggle_neurax_integration.py --test-only
    
    if [ $? -ne 0 ]; then
        log_error "Échec des tests sur les données Kaggle"
        return 1
    fi
    
    log_success "Tests exécutés avec succès sur les données Kaggle"
    return 0
}

# Phase 5: Soumission des résultats à Kaggle
submit_to_kaggle() {
    log_info "$(timestamp) - Phase 5: Soumission des résultats à Kaggle..."
    
    # Exécuter le workflow complet
    python kaggle_neurax_integration.py
    
    if [ $? -ne 0 ]; then
        log_error "Échec de la soumission à Kaggle"
        return 1
    fi
    
    log_success "Résultats soumis avec succès à Kaggle"
    return 0
}

# Fonction principale
main() {
    log_info "Démarrage de la finalisation du projet Neurax2..."
    log_info "Timestamp de démarrage: $(timestamp)"
    
    # Phase 1: Vérification du système
    verify_system
    if [ $? -ne 0 ]; then
        log_error "Finalisation interrompue après la phase 1"
        exit 1
    fi
    
    # Phase 2: Génération des fichiers finaux
    generate_final_files
    if [ $? -ne 0 ]; then
        log_error "Finalisation interrompue après la phase 2"
        exit 1
    fi
    
    # Phase 3: Préparation pour Kaggle
    prepare_kaggle
    if [ $? -ne 0 ]; then
        log_error "Finalisation interrompue après la phase 3"
        exit 1
    fi
    
    # Phase 4: Exécution des tests sur Kaggle
    run_kaggle_tests
    if [ $? -ne 0 ]; then
        log_error "Finalisation interrompue après la phase 4"
        exit 1
    fi
    
    # Phase 5: Soumission des résultats à Kaggle
    submit_to_kaggle
    if [ $? -ne 0 ]; then
        log_error "Finalisation interrompue après la phase 5"
        exit 1
    fi
    
    log_info "Finalisation du projet terminée à: $(timestamp)"
    log_success "Toutes les phases ont été exécutées avec succès"
    log_success "Le projet Neurax2 est maintenant terminé à 100% et validé sur Kaggle!"
}

# Exécuter la fonction principale
main