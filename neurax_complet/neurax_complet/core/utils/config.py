"""
Gestion de la configuration pour le Réseau Neuronal Gravitationnel Quantique
"""

import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class Config:
    """
    Classe de gestion de la configuration.
    Permet de charger, sauvegarder et accéder aux paramètres de configuration.
    """
    
    DEFAULT_CONFIG = {
        # Paramètres généraux
        "node_id": None,  # Sera généré si None
        "log_level": "INFO",
        
        # Paramètres du simulateur quantique
        "quantum": {
            "grid_size": 64,
            "time_steps": 8,
            "default_intensity": 1e-6
        },
        
        # Paramètres neuronaux
        "neuron": {
            "p_0": 0.5,
            "beta_1": 0.3,
            "beta_2": 0.3,
            "beta_3": 0.2,
            "activation_threshold": 0.7
        },
        
        # Paramètres réseau P2P
        "network": {
            "local_port": 5000,
            "bootstrap_nodes": [],
            "max_connections": 50,
            "min_connections": 5,
            "connection_timeout": 30,
            "ping_interval": 60
        },
        
        # Paramètres de consensus
        "consensus": {
            "min_validations": 3,
            "validation_timeout": 300,
            "confidence_threshold": 0.7
        },
        
        # Paramètres UI
        "ui": {
            "theme": "dark",
            "visualization": {
                "update_interval": 500,
                "max_datapoints": 1000
            }
        },
        
        # Paramètres de stockage
        "storage": {
            "data_dir": "data",
            "max_log_size_mb": 100,
            "max_logs": 10,
            "autosave_interval": 300
        }
    }
    
    def __init__(self, config_path=None):
        """
        Initialise la configuration
        
        Args:
            config_path (str): Chemin vers le fichier de configuration.
                              Si None, utilise 'config/default.json'
        """
        self.config_path = config_path or 'config/default.json'
        self.config = dict(self.DEFAULT_CONFIG)  # Copie de la config par défaut
        
        # Charger la configuration depuis le fichier
        self.load()
        
    def load(self):
        """
        Charge la configuration depuis le fichier
        
        Returns:
            bool: True si chargé avec succès, False sinon
        """
        try:
            config_file = Path(self.config_path)
            
            if not config_file.exists():
                logger.warning(f"Config file not found: {self.config_path}")
                return False
                
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
                
            # Mettre à jour la configuration avec les valeurs chargées
            self._update_recursive(self.config, loaded_config)
            
            logger.info(f"Configuration loaded from {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return False
            
    def save(self):
        """
        Sauvegarde la configuration dans le fichier
        
        Returns:
            bool: True si sauvegardé avec succès, False sinon
        """
        try:
            config_file = Path(self.config_path)
            
            # Créer le répertoire parent si nécessaire
            os.makedirs(config_file.parent, exist_ok=True)
            
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
                
            logger.info(f"Configuration saved to {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
            
    def get(self, key, default=None):
        """
        Obtient une valeur de configuration
        
        Args:
            key (str): Clé de configuration, peut être imbriquée avec des points
                      (ex: "network.local_port")
            default: Valeur par défaut si la clé n'existe pas
            
        Returns:
            Valeur de configuration ou default si non trouvée
        """
        parts = key.split('.')
        config = self.config
        
        try:
            for part in parts:
                config = config[part]
            return config
        except (KeyError, TypeError):
            return default
            
    def set(self, key, value):
        """
        Définit une valeur de configuration
        
        Args:
            key (str): Clé de configuration, peut être imbriquée avec des points
            value: Valeur à définir
            
        Returns:
            bool: True si défini avec succès, False sinon
        """
        parts = key.split('.')
        config = self.config
        
        try:
            # Naviguer jusqu'au parent de la clé à définir
            for part in parts[:-1]:
                if part not in config:
                    config[part] = {}
                config = config[part]
                
            # Définir la valeur
            config[parts[-1]] = value
            return True
            
        except Exception as e:
            logger.error(f"Error setting configuration value: {e}")
            return False
            
    def _update_recursive(self, target, source):
        """
        Met à jour récursivement un dictionnaire
        
        Args:
            target (dict): Dictionnaire cible à mettre à jour
            source (dict): Dictionnaire source avec nouvelles valeurs
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Mise à jour récursive pour les sous-dictionnaires
                self._update_recursive(target[key], value)
            else:
                # Mise à jour directe pour les autres types
                target[key] = value
                
    def reset_to_defaults(self):
        """
        Réinitialise la configuration aux valeurs par défaut
        
        Returns:
            bool: True si réinitialisé avec succès, False sinon
        """
        try:
            self.config = dict(self.DEFAULT_CONFIG)
            return True
        except Exception as e:
            logger.error(f"Error resetting configuration: {e}")
            return False
            
    def to_dict(self):
        """
        Convertit la configuration en dictionnaire
        
        Returns:
            dict: Copie de la configuration
        """
        return dict(self.config)