"""
Définition des formats de messages et des fonctions de sérialisation/désérialisation
pour la communication P2P dans le réseau neuronal gravitationnel quantique.
"""

import json
import time
import uuid
import base64
import hashlib
import logging
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class MessageType:
    """Types de messages pour le protocole NeuralMesh"""
    HELLO = "HELLO"            # Découverte et présentation
    STATE = "STATE"            # Partage d'état neuronal
    PROBLEM = "PROBLEM"        # Soumission d'un problème
    SOLUTION = "SOLUTION"      # Proposition de solution
    VALIDATION = "VALIDATION"  # Confirmation ou contestation
    KNOWLEDGE = "KNOWLEDGE"    # Partage de connaissance validée
    QUERY = "QUERY"            # Demande d'information
    PEER_LIST = "PEER_LIST"    # Liste de pairs connus
    PING = "PING"              # Vérification de connectivité
    PONG = "PONG"              # Réponse de connectivité

class Message:
    """
    Classe représentant un message dans le protocole NeuralMesh.
    Gère la sérialisation, désérialisation et validation des messages.
    """
    
    PROTOCOL_VERSION = "1.0.0"
    
    def __init__(self, msg_type, sender_id, content=None, msg_id=None, timestamp=None, 
                 references=None, signature=None):
        """
        Initialise un nouveau message.
        
        Args:
            msg_type (str): Type de message (voir MessageType)
            sender_id (str): Identifiant du nœud émetteur
            content (dict): Contenu du message (spécifique au type)
            msg_id (str): Identifiant unique du message (généré si None)
            timestamp (float): Horodatage du message (généré si None)
            references (list): Liste d'identifiants de messages référencés
            signature (str): Signature cryptographique du message
        """
        self.type = msg_type
        self.sender_id = sender_id
        self.content = content or {}
        self.msg_id = msg_id or str(uuid.uuid4())
        self.timestamp = timestamp or time.time()
        self.references = references or []
        self.signature = signature
        self.version = self.PROTOCOL_VERSION
        
    def to_dict(self):
        """
        Convertit le message en dictionnaire.
        
        Returns:
            dict: Représentation dictionnaire du message
        """
        return {
            "msg_id": self.msg_id,
            "type": self.type,
            "sender_id": self.sender_id,
            "timestamp": self.timestamp,
            "version": self.version,
            "content": self.content,
            "references": self.references,
            "signature": self.signature
        }
        
    def to_json(self):
        """
        Sérialise le message en JSON.
        
        Returns:
            str: Représentation JSON du message
        """
        return json.dumps(self.to_dict())
        
    def sign(self, private_key):
        """
        Signe le message avec la clé privée fournie.
        
        Args:
            private_key: Clé privée pour la signature
            
        Note: Cette méthode devra être implémentée avec une vraie
        bibliothèque cryptographique dans l'implémentation finale.
        """
        # Structure à signer (sans la signature elle-même)
        to_sign = {
            "msg_id": self.msg_id,
            "type": self.type,
            "sender_id": self.sender_id,
            "timestamp": self.timestamp,
            "version": self.version,
            "content": self.content,
            "references": self.references
        }
        
        # Conversion en chaîne canonique
        canonical = json.dumps(to_sign, sort_keys=True)
        
        # Pour la démonstration, utilisation de hash comme signature
        # Dans l'implémentation finale, remplacer par signature cryptographique réelle
        hash_value = hashlib.sha256((canonical + str(private_key)).encode()).digest()
        self.signature = base64.b64encode(hash_value).decode()
        
        return self.signature
        
    def verify(self, public_key):
        """
        Vérifie la signature du message avec la clé publique fournie.
        
        Args:
            public_key: Clé publique pour la vérification
            
        Returns:
            bool: True si la signature est valide, False sinon
            
        Note: Cette méthode devra être implémentée avec une vraie
        bibliothèque cryptographique dans l'implémentation finale.
        """
        if not self.signature:
            logger.warning("Message has no signature")
            return False
            
        # Structure signée (sans la signature elle-même)
        signed_data = {
            "msg_id": self.msg_id,
            "type": self.type,
            "sender_id": self.sender_id,
            "timestamp": self.timestamp,
            "version": self.version,
            "content": self.content,
            "references": self.references
        }
        
        # Conversion en chaîne canonique
        canonical = json.dumps(signed_data, sort_keys=True)
        
        # Pour la démonstration, vérification du hash comme signature
        # Dans l'implémentation finale, remplacer par vérification cryptographique réelle
        expected_hash = hashlib.sha256((canonical + str(public_key)).encode()).digest()
        expected_signature = base64.b64encode(expected_hash).decode()
        
        return self.signature == expected_signature
        
    @classmethod
    def from_dict(cls, data):
        """
        Crée un message à partir d'un dictionnaire.
        
        Args:
            data (dict): Dictionnaire représentant le message
            
        Returns:
            Message: Instance de Message créée
        """
        return cls(
            msg_type=data.get("type"),
            sender_id=data.get("sender_id"),
            content=data.get("content", {}),
            msg_id=data.get("msg_id"),
            timestamp=data.get("timestamp"),
            references=data.get("references", []),
            signature=data.get("signature")
        )
        
    @classmethod
    def from_json(cls, json_str):
        """
        Crée un message à partir d'une chaîne JSON.
        
        Args:
            json_str (str): Chaîne JSON représentant le message
            
        Returns:
            Message: Instance de Message créée
        """
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding message JSON: {e}")
            return None
            
    @classmethod
    def create_hello(cls, sender_id, capabilities=None, public_key=None):
        """
        Crée un message HELLO pour l'établissement initial de connexion.
        
        Args:
            sender_id (str): Identifiant du nœud émetteur
            capabilities (dict): Capacités et caractéristiques du nœud
            public_key (str): Clé publique du nœud pour authentification
            
        Returns:
            Message: Message HELLO créé
        """
        content = {
            "protocol_version": cls.PROTOCOL_VERSION,
            "node_time": datetime.now().isoformat(),
            "capabilities": capabilities or {},
            "public_key": public_key
        }
        
        return cls(
            msg_type=MessageType.HELLO,
            sender_id=sender_id,
            content=content
        )
        
    @classmethod
    def create_state(cls, sender_id, neuron_state):
        """
        Crée un message STATE pour partager l'état neuronal.
        
        Args:
            sender_id (str): Identifiant du nœud émetteur
            neuron_state (dict): État du neurone à partager
            
        Returns:
            Message: Message STATE créé
        """
        # Nettoyer l'état pour transmission (éviter les grands tableaux)
        clean_state = neuron_state.copy()
        if 'space_time_data' in clean_state:
            del clean_state['space_time_data']
            
        content = {
            "neuron_state": clean_state,
            "timestamp": datetime.now().isoformat()
        }
        
        return cls(
            msg_type=MessageType.STATE,
            sender_id=sender_id,
            content=content
        )
        
    @classmethod
    def create_problem(cls, sender_id, problem_data, problem_type, problem_id=None):
        """
        Crée un message PROBLEM pour soumettre un problème au réseau.
        
        Args:
            sender_id (str): Identifiant du nœud émetteur
            problem_data (dict): Données du problème
            problem_type (str): Type de problème
            problem_id (str): Identifiant du problème (généré si None)
            
        Returns:
            Message: Message PROBLEM créé
        """
        problem_id = problem_id or str(uuid.uuid4())
        
        content = {
            "problem_id": problem_id,
            "problem_type": problem_type,
            "problem_data": problem_data,
            "timestamp": datetime.now().isoformat(),
            "difficulty_estimate": estimate_problem_complexity(problem_data, problem_type)
        }
        
        return cls(
            msg_type=MessageType.PROBLEM,
            sender_id=sender_id,
            content=content
        )
        
    @classmethod
    def create_solution(cls, sender_id, problem_id, solution_data, confidence=None):
        """
        Crée un message SOLUTION pour proposer une solution à un problème.
        
        Args:
            sender_id (str): Identifiant du nœud émetteur
            problem_id (str): Identifiant du problème résolu
            solution_data (dict): Données de la solution
            confidence (float): Niveau de confiance dans la solution (0-1)
            
        Returns:
            Message: Message SOLUTION créé
        """
        solution_id = str(uuid.uuid4())
        
        content = {
            "solution_id": solution_id,
            "problem_id": problem_id,
            "solution_data": solution_data,
            "confidence": confidence or 1.0,
            "timestamp": datetime.now().isoformat(),
            "computation_time": 0  # À remplir par l'appelant si nécessaire
        }
        
        return cls(
            msg_type=MessageType.SOLUTION,
            sender_id=sender_id,
            content=content
        )
        
    @classmethod
    def create_validation(cls, sender_id, solution_id, is_valid, validation_reason=None):
        """
        Crée un message VALIDATION pour confirmer ou rejeter une solution.
        
        Args:
            sender_id (str): Identifiant du nœud émetteur
            solution_id (str): Identifiant de la solution validée
            is_valid (bool): True si la solution est valide, False sinon
            validation_reason (str): Raison de la validation/rejet
            
        Returns:
            Message: Message VALIDATION créé
        """
        content = {
            "solution_id": solution_id,
            "is_valid": is_valid,
            "validation_reason": validation_reason,
            "timestamp": datetime.now().isoformat()
        }
        
        return cls(
            msg_type=MessageType.VALIDATION,
            sender_id=sender_id,
            content=content
        )
        
    @classmethod
    def create_knowledge(cls, sender_id, knowledge_data, knowledge_type, references=None):
        """
        Crée un message KNOWLEDGE pour partager une connaissance validée.
        
        Args:
            sender_id (str): Identifiant du nœud émetteur
            knowledge_data (dict): Données de la connaissance
            knowledge_type (str): Type de connaissance
            references (list): Liste d'identifiants de messages référencés
            
        Returns:
            Message: Message KNOWLEDGE créé
        """
        knowledge_id = str(uuid.uuid4())
        
        content = {
            "knowledge_id": knowledge_id,
            "knowledge_type": knowledge_type,
            "knowledge_data": knowledge_data,
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "origin": sender_id,
                "creation_time": datetime.now().isoformat()
            }
        }
        
        return cls(
            msg_type=MessageType.KNOWLEDGE,
            sender_id=sender_id,
            content=content,
            references=references or []
        )
        
    @classmethod
    def create_query(cls, sender_id, query_type, query_params=None):
        """
        Crée un message QUERY pour demander des informations.
        
        Args:
            sender_id (str): Identifiant du nœud émetteur
            query_type (str): Type de requête
            query_params (dict): Paramètres de la requête
            
        Returns:
            Message: Message QUERY créé
        """
        query_id = str(uuid.uuid4())
        
        content = {
            "query_id": query_id,
            "query_type": query_type,
            "query_params": query_params or {},
            "timestamp": datetime.now().isoformat(),
            "ttl": 10  # Time-to-live en sauts réseau
        }
        
        return cls(
            msg_type=MessageType.QUERY,
            sender_id=sender_id,
            content=content
        )
        
    @classmethod
    def create_peer_list(cls, sender_id, peers):
        """
        Crée un message PEER_LIST pour partager des pairs connus.
        
        Args:
            sender_id (str): Identifiant du nœud émetteur
            peers (list): Liste de pairs connus avec leurs informations
            
        Returns:
            Message: Message PEER_LIST créé
        """
        content = {
            "peers": peers,
            "timestamp": datetime.now().isoformat()
        }
        
        return cls(
            msg_type=MessageType.PEER_LIST,
            sender_id=sender_id,
            content=content
        )
        
    @classmethod
    def create_ping(cls, sender_id):
        """
        Crée un message PING pour vérifier la connectivité.
        
        Args:
            sender_id (str): Identifiant du nœud émetteur
            
        Returns:
            Message: Message PING créé
        """
        content = {
            "timestamp": datetime.now().isoformat()
        }
        
        return cls(
            msg_type=MessageType.PING,
            sender_id=sender_id,
            content=content
        )
        
    @classmethod
    def create_pong(cls, sender_id, ping_id):
        """
        Crée un message PONG en réponse à un PING.
        
        Args:
            sender_id (str): Identifiant du nœud émetteur
            ping_id (str): Identifiant du message PING
            
        Returns:
            Message: Message PONG créé
        """
        content = {
            "timestamp": datetime.now().isoformat(),
            "round_trip_start": time.time()  # Pour mesurer le RTT
        }
        
        return cls(
            msg_type=MessageType.PONG,
            sender_id=sender_id,
            content=content,
            references=[ping_id]
        )


def estimate_problem_complexity(problem_data, problem_type):
    """
    Estime la complexité d'un problème pour guider l'allocation de ressources.
    
    Args:
        problem_data (dict): Données du problème
        problem_type (str): Type de problème
        
    Returns:
        float: Estimation de la complexité entre 0 et 1
    """
    # Implémentation simple basée sur la taille des données
    # À étendre avec des heuristiques spécifiques aux types de problèmes
    complexity = 0.5  # Complexité moyenne par défaut
    
    try:
        # Estimer en fonction du type de problème
        if problem_type == "OPTIMIZATION":
            # Évaluer la dimensionnalité et les contraintes
            dimensions = problem_data.get("dimensions", 1)
            constraints = len(problem_data.get("constraints", []))
            complexity = min(0.2 + 0.1 * dimensions + 0.05 * constraints, 1.0)
            
        elif problem_type == "SIMULATION":
            # Évaluer la taille de la simulation et le nombre d'étapes
            grid_size = problem_data.get("grid_size", 50)
            time_steps = problem_data.get("time_steps", 100)
            complexity = min(0.3 + (grid_size/100)**2 * 0.3 + (time_steps/500) * 0.4, 1.0)
            
        elif problem_type == "PATTERN_RECOGNITION":
            # Évaluer la taille des données et la complexité du motif
            data_points = len(problem_data.get("data", []))
            features = problem_data.get("features", 1)
            complexity = min(0.2 + (data_points/10000) * 0.4 + (features/20) * 0.4, 1.0)
            
        elif problem_type == "CREATIVE":
            # Les problèmes créatifs sont difficiles à quantifier
            constraints = len(problem_data.get("constraints", []))
            dimensions = problem_data.get("dimensions", 1)
            complexity = min(0.6 + 0.02 * constraints + 0.02 * dimensions, 1.0)
            
    except Exception as e:
        logger.warning(f"Error estimating problem complexity: {e}")
        
    return complexity