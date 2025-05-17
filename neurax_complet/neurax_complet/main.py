#!/usr/bin/env python3
"""
Réseau Neuronal Gravitationnel Quantique Décentralisé
Point d'entrée principal de l'application
"""

import os
import sys
import logging
import argparse
import asyncio
import threading
import time
import subprocess
from datetime import datetime
from pathlib import Path

# Ajout des chemins pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# S'assurer que les répertoires nécessaires existent
os.makedirs('logs', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('config', exist_ok=True)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/quantum_network_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)

logger = logging.getLogger("QuantumNetwork")

def parse_arguments():
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(description="Réseau Neuronal Gravitationnel Quantique Décentralisé")
    
    parser.add_argument('--mode', choices=['full', 'quantum', 'p2p', 'ui'], default='full',
                        help='Mode de démarrage: full (complet), quantum (simulation uniquement), '
                             'p2p (réseau uniquement), ui (interface uniquement)')
    
    parser.add_argument('--config', type=str, default='config/default.json',
                        help='Chemin vers le fichier de configuration')
    
    parser.add_argument('--debug', action='store_true',
                        help='Active le mode debug avec logging verbeux')
    
    parser.add_argument('--bootstrap', type=str, 
                        help='Adresse du nœud bootstrap pour rejoindre le réseau (format: ip:port)')
    
    parser.add_argument('--port', type=int, default=5000,
                        help='Port d\'écoute pour les connections P2P')
    
    parser.add_argument('--streamlit', action='store_true',
                        help='Démarre l\'interface Streamlit')
    
    return parser.parse_args()

def setup_environment(args):
    """Configure l'environnement d'exécution"""
    # Ajuster le niveau de logging si mode debug
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Mode debug activé")

async def run_p2p_network(port, bootstrap):
    """Lance le réseau P2P"""
    try:
        # Import et initialisation du réseau P2P
        from core.p2p.network import P2PNetwork
        
        # Créer le réseau
        network = P2PNetwork(local_port=port, bootstrap=bootstrap)
        
        # Démarrer le réseau
        await network.start()
        
        logger.info(f"Réseau P2P démarré sur le port {port}")
        
        # Maintenir la connexion active
        while True:
            await asyncio.sleep(1)
            
    except Exception as e:
        logger.error(f"Erreur réseau P2P: {str(e)}", exc_info=True)

def run_quantum_simulator():
    """Lance le simulateur quantique"""
    try:
        # Import et initialisation du moteur quantique
        from core.quantum_sim.simulator import QuantumGravitySimulator
        from core.neuron.quantum_neuron import QuantumGravitationalNeuron
        
        # Charger la configuration
        from core.utils.config import Config
        config = Config()
        
        # Créer le simulateur
        neuron = QuantumGravitationalNeuron(
            grid_size=config.get("quantum.grid_size", 64),
            time_steps=config.get("quantum.time_steps", 8),
            p_0=config.get("neuron.p_0", 0.5),
            beta_1=config.get("neuron.beta_1", 0.3),
            beta_2=config.get("neuron.beta_2", 0.3),
            beta_3=config.get("neuron.beta_3", 0.2)
        )
        
        logger.info("Neurone quantique gravitationnel initialisé")
        
        # Boucle de simulation
        intensity = config.get("quantum.default_intensity", 1e-6)
        while True:
            neuron.neuron_step(intensity)
            time.sleep(0.1)
            
    except Exception as e:
        logger.error(f"Erreur simulateur quantique: {str(e)}", exc_info=True)

def run_streamlit():
    """Lance l'interface Streamlit"""
    try:
        logger.info("Démarrage de l'interface Streamlit...")
        
        streamlit_path = Path("ui/web/app.py").absolute()
        
        # Vérifier que le fichier existe
        if not streamlit_path.exists():
            logger.error(f"Fichier Streamlit introuvable: {streamlit_path}")
            return
            
        # Lancer Streamlit
        process = subprocess.Popen(
            ["streamlit", "run", str(streamlit_path), "--server.port=5000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        logger.info(f"Interface Streamlit démarrée (PID: {process.pid})")
        
        # Surveiller le processus Streamlit
        while True:
            output = process.stdout.readline()
            if output:
                logger.debug(f"Streamlit: {output.strip()}")
                
            if process.poll() is not None:
                logger.info("Streamlit s'est arrêté")
                break
                
            time.sleep(0.1)
            
    except Exception as e:
        logger.error(f"Erreur démarrage Streamlit: {str(e)}", exc_info=True)

def main():
    """Fonction principale"""
    # Analyse des arguments
    args = parse_arguments()
    
    # Configuration de l'environnement
    setup_environment(args)
    
    logger.info(f"Démarrage du Réseau Neuronal Gravitationnel Quantique en mode {args.mode}")
    
    try:
        threads = []
        
        # Démarrer les composants selon le mode
        if args.mode in ['full', 'quantum']:
            # Lancer le simulateur quantique dans un thread séparé
            simulator_thread = threading.Thread(target=run_quantum_simulator, daemon=True)
            simulator_thread.start()
            threads.append(simulator_thread)
            
        if args.mode in ['full', 'p2p']:
            # Lancer le réseau P2P dans une tâche asyncio
            # Configurer la boucle asyncio dans un thread séparé
            async def run_network():
                await run_p2p_network(args.port, args.bootstrap)
                
            def network_thread_func():
                asyncio.run(run_network())
                
            network_thread = threading.Thread(target=network_thread_func, daemon=True)
            network_thread.start()
            threads.append(network_thread)
            
        if args.mode in ['full', 'ui'] or args.streamlit:
            # Lancer Streamlit directement
            run_streamlit()
            
        # Attendre l'interruption de l'utilisateur
        for thread in threads:
            thread.join()
            
    except KeyboardInterrupt:
        logger.info("Arrêt manuel du programme")
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution: {str(e)}", exc_info=True)
    finally:
        logger.info("Arrêt du Réseau Neuronal Gravitationnel Quantique")
        
if __name__ == "__main__":
    main()