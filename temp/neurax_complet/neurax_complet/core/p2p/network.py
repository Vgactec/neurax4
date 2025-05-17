"""
Infrastructure réseau P2P pour le Réseau Neuronal Gravitationnel Quantique
Ce module implémente la couche de communication décentralisée
"""

import os
import socket
import threading
import time
import json
import logging
import random
import asyncio
import hashlib
import uuid
from datetime import datetime, timedelta

from .message import Message, MessageType
from ..utils.config import Config

logger = logging.getLogger(__name__)

class PeerInfo:
    """
    Classe contenant les informations sur un pair dans le réseau
    """
    
    def __init__(self, node_id, address, port, public_key=None, last_seen=None, capabilities=None):
        """
        Initialise les informations d'un pair
        
        Args:
            node_id (str): Identifiant unique du pair
            address (str): Adresse IP ou hostname du pair
            port (int): Port d'écoute du pair
            public_key (str): Clé publique du pair pour authentification
            last_seen (float): Timestamp de dernière connexion
            capabilities (dict): Capacités et caractéristiques du pair
        """
        self.node_id = node_id
        self.address = address
        self.port = port
        self.public_key = public_key
        self.last_seen = last_seen or time.time()
        self.capabilities = capabilities or {}
        self.connection_attempts = 0
        self.connection_successes = 0
        self.reputation = 0.5  # Score de réputation initial (0-1)
        self.latency_history = []  # Historique des latences
        self.connection_active = False
        
    def get_endpoint(self):
        """
        Renvoie l'adresse endpoint complète du pair
        
        Returns:
            str: Adresse au format "address:port"
        """
        return f"{self.address}:{self.port}"
        
    def update_last_seen(self):
        """Met à jour le timestamp de dernière activité"""
        self.last_seen = time.time()
        
    def update_latency(self, latency_ms):
        """
        Ajoute une mesure de latence à l'historique
        
        Args:
            latency_ms (float): Latence mesurée en millisecondes
        """
        self.latency_history.append(latency_ms)
        if len(self.latency_history) > 20:
            self.latency_history.pop(0)
            
    def get_average_latency(self):
        """
        Calcule la latence moyenne
        
        Returns:
            float: Latence moyenne en millisecondes ou 0 si aucune mesure
        """
        if not self.latency_history:
            return 0
        return sum(self.latency_history) / len(self.latency_history)
        
    def update_reputation(self, change):
        """
        Met à jour le score de réputation
        
        Args:
            change (float): Changement à appliquer (-1 à 1)
        """
        self.reputation = max(0, min(1, self.reputation + change))
        
    def is_active(self, max_inactive_seconds=300):
        """
        Vérifie si le pair est considéré comme actif
        
        Args:
            max_inactive_seconds (int): Temps maximum d'inactivité en secondes
            
        Returns:
            bool: True si le pair est actif, False sinon
        """
        return (time.time() - self.last_seen) < max_inactive_seconds
        
    def to_dict(self):
        """
        Convertit les informations du pair en dictionnaire
        
        Returns:
            dict: Représentation dictionnaire des informations du pair
        """
        return {
            "node_id": self.node_id,
            "address": self.address,
            "port": self.port,
            "public_key": self.public_key,
            "last_seen": self.last_seen,
            "capabilities": self.capabilities,
            "reputation": self.reputation,
            "avg_latency": self.get_average_latency(),
            "active": self.connection_active
        }
        
    @classmethod
    def from_dict(cls, data):
        """
        Crée une instance PeerInfo à partir d'un dictionnaire
        
        Args:
            data (dict): Dictionnaire contenant les informations du pair
            
        Returns:
            PeerInfo: Instance créée
        """
        peer = cls(
            node_id=data.get("node_id"),
            address=data.get("address"),
            port=data.get("port"),
            public_key=data.get("public_key"),
            last_seen=data.get("last_seen"),
            capabilities=data.get("capabilities", {})
        )
        
        peer.reputation = data.get("reputation", 0.5)
        peer.connection_active = data.get("active", False)
        
        return peer


class Connection:
    """
    Gère une connexion active avec un pair
    """
    
    def __init__(self, peer_info, socket=None):
        """
        Initialise une connexion
        
        Args:
            peer_info (PeerInfo): Informations sur le pair
            socket: Socket connecté ou None
        """
        self.peer_info = peer_info
        self.socket = socket
        self.send_queue = asyncio.Queue()
        self.active = False
        self.last_ping_time = 0
        self.last_ping_id = None
        self.lock = threading.Lock()
        
    async def send_message(self, message):
        """
        Envoie un message de façon asynchrone
        
        Args:
            message (Message): Message à envoyer
            
        Returns:
            bool: True si envoyé avec succès, False sinon
        """
        if not self.active:
            logger.warning(f"Cannot send message to inactive connection {self.peer_info.node_id}")
            return False
            
        try:
            await self.send_queue.put(message)
            return True
        except Exception as e:
            logger.error(f"Error queuing message: {e}")
            return False
            
    async def _send_worker(self):
        """Worker pour envoyer les messages de la queue"""
        while self.active:
            try:
                message = await self.send_queue.get()
                json_data = message.to_json()
                data = json_data.encode() + b'\n'  # Délimiteur pour message framing
                
                # Mettre une longueur en tête pour faciliter la lecture côté réception
                length = len(data).to_bytes(4, byteorder='big')
                
                with self.lock:
                    if self.socket:
                        await asyncio.get_event_loop().sock_sendall(self.socket, length + data)
                        logger.debug(f"Sent {message.type} message to {self.peer_info.node_id}")
                    else:
                        logger.warning(f"Socket not available for {self.peer_info.node_id}")
                
                self.send_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in send worker: {e}")
                await asyncio.sleep(1)  # Éviter de spammer en cas d'erreur
                
    async def start(self):
        """Démarre la connexion"""
        self.active = True
        asyncio.create_task(self._send_worker())
        
    async def close(self):
        """Ferme la connexion"""
        self.active = False
        
        with self.lock:
            if self.socket:
                try:
                    self.socket.close()
                except Exception as e:
                    logger.error(f"Error closing socket: {e}")
                finally:
                    self.socket = None
                
    async def ping(self):
        """
        Envoie un ping au pair pour vérifier la connexion
        
        Returns:
            str: ID du message ping
        """
        message = Message.create_ping("local_node_id")  # À remplacer par l'ID local réel
        self.last_ping_time = time.time()
        self.last_ping_id = message.msg_id
        
        await self.send_message(message)
        return message.msg_id


class P2PNetwork:
    """
    Implémentation du réseau P2P pour le Réseau Neuronal Gravitationnel Quantique
    """
    
    def __init__(self, local_port=5000, bootstrap=None, node_id=None, 
                 max_connections=50, min_connections=5):
        """
        Initialise le réseau P2P
        
        Args:
            local_port (int): Port d'écoute local
            bootstrap (str): Adresse "host:port" pour bootstrap initial
            node_id (str): ID du nœud local (généré si None)
            max_connections (int): Nombre maximum de connexions simultanées
            min_connections (int): Nombre minimum de connexions à maintenir
        """
        self.logger = logging.getLogger(__name__)
        self.port = local_port  # Fix: Initialize port attribute
        
        # Identifiant unique pour ce nœud
        self.node_id = node_id or self._generate_node_id()
        
        # Configuration réseau
        self.port = local_port
        self.bootstrap_nodes = []
        if bootstrap:
            host, port = bootstrap.split(":")
            self.bootstrap_nodes.append((host, int(port)))
            
        self.max_connections = max_connections
        self.min_connections = min_connections
        
        # Gestion des pairs
        self.peers = {}  # {node_id: PeerInfo}
        self.connections = {}  # {node_id: Connection}
        self.blacklist = set()  # Ensemble de node_ids blacklistés
        
        # Traitement des messages
        self.message_handlers = {
            MessageType.HELLO: self._handle_hello,
            MessageType.STATE: self._handle_state,
            MessageType.PROBLEM: self._handle_problem,
            MessageType.SOLUTION: self._handle_solution,
            MessageType.VALIDATION: self._handle_validation,
            MessageType.KNOWLEDGE: self._handle_knowledge,
            MessageType.QUERY: self._handle_query,
            MessageType.PEER_LIST: self._handle_peer_list,
            MessageType.PING: self._handle_ping,
            MessageType.PONG: self._handle_pong
        }
        
        # Hooks pour intégration externe
        self.on_peer_connected = None
        self.on_peer_disconnected = None
        self.on_state_received = None
        self.on_problem_received = None
        self.on_solution_received = None
        
        # État du réseau
        self.running = False
        self.server_socket = None
        self.accept_task = None
        self.maintenance_task = None
        
        self.logger.info(f"P2P Network initialized with node ID: {self.node_id}")
        self.logger.info(f"Listening on port {self.port}")
        
    def _generate_node_id(self):
        """
        Génère un identifiant unique pour ce nœud
        
        Returns:
            str: Identifiant unique
        """
        # Combiner plusieurs sources d'entropie pour l'unicité
        unique_data = {
            'timestamp': time.time(),
            'random': random.random(),
            'hostname': socket.gethostname(),
            'port': self.port
        }
        
        # Créer un hash unique
        serialized = json.dumps(unique_data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]
        
    async def start(self):
        """
        Démarre le réseau P2P
        
        Returns:
            bool: True si démarré avec succès, False sinon
        """
        if self.running:
            self.logger.warning("P2P Network already running")
            return False
            
        try:
            # Créer et configurer le socket serveur
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('0.0.0.0', self.port))
            self.server_socket.listen(100)
            self.server_socket.setblocking(False)
            
            self.running = True
            
            # Démarrer les tâches asynchrones
            self.accept_task = asyncio.create_task(self._accept_connections())
            self.maintenance_task = asyncio.create_task(self._network_maintenance())
            
            # Connexion aux nœuds bootstrap
            if self.bootstrap_nodes:
                for host, port in self.bootstrap_nodes:
                    asyncio.create_task(self._connect_to_bootstrap(host, port))
                    
            self.logger.info("P2P Network started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start P2P Network: {e}")
            self.stop()
            return False
            
    async def stop(self):
        """
        Arrête le réseau P2P
        
        Returns:
            bool: True si arrêté avec succès, False sinon
        """
        self.logger.info("Stopping P2P Network...")
        self.running = False
        
        # Annuler les tâches asynchrones
        if self.accept_task:
            self.accept_task.cancel()
            self.accept_task = None
            
        if self.maintenance_task:
            self.maintenance_task.cancel()
            self.maintenance_task = None
            
        # Fermer toutes les connexions
        for connection in list(self.connections.values()):
            await connection.close()
        self.connections.clear()
        
        # Fermer le socket serveur
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception as e:
                self.logger.error(f"Error closing server socket: {e}")
            finally:
                self.server_socket = None
                
        self.logger.info("P2P Network stopped")
        return True
        
    async def _accept_connections(self):
        """Tâche pour accepter les connexions entrantes"""
        self.logger.info("Started accepting connections")
        
        while self.running:
            try:
                client_socket, client_address = await asyncio.get_event_loop().sock_accept(self.server_socket)
                client_socket.setblocking(False)
                
                self.logger.info(f"Accepted connection from {client_address[0]}:{client_address[1]}")
                
                # Créer un identifiant temporaire jusqu'à la réception du HELLO
                temp_id = f"temp_{uuid.uuid4().hex[:8]}"
                
                # Créer une entrée temporaire pour ce pair
                peer_info = PeerInfo(
                    node_id=temp_id,
                    address=client_address[0],
                    port=client_address[1]
                )
                
                # Établir la connexion
                connection = Connection(peer_info, client_socket)
                await connection.start()
                
                # Stocker temporairement
                self.connections[temp_id] = connection
                
                # Lancer une tâche pour gérer cette connexion
                asyncio.create_task(self._handle_connection(connection))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error accepting connection: {e}")
                await asyncio.sleep(1)
                
        self.logger.info("Stopped accepting connections")
        
    async def _handle_connection(self, connection):
        """
        Gère une connexion active
        
        Args:
            connection (Connection): Connexion à gérer
        """
        buffer = b''
        
        while self.running and connection.active:
            try:
                # Lire la taille du message
                while len(buffer) < 4:
                    chunk = await asyncio.get_event_loop().sock_recv(connection.socket, 4096)
                    if not chunk:
                        raise ConnectionError("Connection closed by peer")
                    buffer += chunk
                    
                msg_len = int.from_bytes(buffer[:4], byteorder='big')
                buffer = buffer[4:]
                
                # Lire le reste du message
                while len(buffer) < msg_len:
                    chunk = await asyncio.get_event_loop().sock_recv(connection.socket, 4096)
                    if not chunk:
                        raise ConnectionError("Connection closed by peer")
                    buffer += chunk
                    
                # Extraire et traiter le message
                msg_data = buffer[:msg_len]
                buffer = buffer[msg_len:]
                
                # Décoder et traiter le message
                try:
                    msg_str = msg_data.decode().strip()
                    message = Message.from_json(msg_str)
                    
                    if message:
                        # Pour le premier message, c'est normalement un HELLO
                        if connection.peer_info.node_id.startswith("temp_") and message.type == MessageType.HELLO:
                            # Mettre à jour les informations du pair
                            real_node_id = message.sender_id
                            
                            # Vérifier si on connaît déjà ce pair
                            if real_node_id in self.peers:
                                # Mettre à jour les infos existantes
                                existing_peer = self.peers[real_node_id]
                                existing_peer.address = connection.peer_info.address
                                existing_peer.port = message.content.get("port", connection.peer_info.port)
                                existing_peer.public_key = message.content.get("public_key")
                                existing_peer.capabilities = message.content.get("capabilities", {})
                                existing_peer.update_last_seen()
                                
                                peer_info = existing_peer
                            else:
                                # Créer une nouvelle entrée pour ce pair
                                peer_info = PeerInfo(
                                    node_id=real_node_id,
                                    address=connection.peer_info.address,
                                    port=message.content.get("port", connection.peer_info.port),
                                    public_key=message.content.get("public_key"),
                                    capabilities=message.content.get("capabilities", {})
                                )
                                self.peers[real_node_id] = peer_info
                                
                            # Mettre à jour la connexion
                            del self.connections[connection.peer_info.node_id]
                            connection.peer_info = peer_info
                            self.connections[real_node_id] = connection
                            peer_info.connection_active = True
                            
                            self.logger.info(f"Identified peer as {real_node_id}")
                            
                            # Appeler le hook si défini
                            if self.on_peer_connected:
                                asyncio.create_task(self.on_peer_connected(real_node_id, peer_info))
                                
                            # Envoyer un HELLO en retour si c'était le premier
                            our_hello = Message.create_hello(
                                sender_id=self.node_id,
                                capabilities={"type": "full_node"},
                                public_key="our_public_key"  # À remplacer par la vraie clé
                            )
                            await connection.send_message(our_hello)
                            
                            # Envoyer notre liste de pairs connus
                            await self._send_peer_list(connection)
                            
                        # Traiter le message selon son type
                        await self._process_message(message, connection)
                        
                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid message format: {e}")
                except Exception as e:
                    self.logger.error(f"Error processing message: {e}")
                    
            except ConnectionError as e:
                self.logger.info(f"Connection closed: {e}")
                break
            except Exception as e:
                self.logger.error(f"Connection error: {e}")
                break
                
        # Fermer la connexion
        await self._disconnect_peer(connection.peer_info.node_id)
        
    async def _process_message(self, message, connection):
        """
        Traite un message reçu
        
        Args:
            message (Message): Message à traiter
            connection (Connection): Connexion source
        """
        # Mettre à jour le timestamp de dernière activité
        connection.peer_info.update_last_seen()
        
        # Vérifier la signature (si implémentée)
        # if not message.verify(connection.peer_info.public_key):
        #     self.logger.warning(f"Invalid message signature from {connection.peer_info.node_id}")
        #     return
        
        # Appeler le handler approprié
        if message.type in self.message_handlers:
            try:
                await self.message_handlers[message.type](message, connection)
            except Exception as e:
                self.logger.error(f"Error in message handler for {message.type}: {e}")
        else:
            self.logger.warning(f"Unsupported message type: {message.type}")
            
    async def _handle_hello(self, message, connection):
        """Traite un message HELLO"""
        self.logger.debug(f"Received HELLO from {message.sender_id}")
        
        # Mise à jour des informations de capacité du pair
        if 'capabilities' in message.content:
            connection.peer_info.capabilities = message.content['capabilities']
            
        # Mise à jour de la clé publique si disponible
        if 'public_key' in message.content:
            connection.peer_info.public_key = message.content['public_key']
            
    async def _handle_state(self, message, connection):
        """Traite un message STATE"""
        self.logger.debug(f"Received STATE from {message.sender_id}")
        
        # Si le hook est défini, transmettre l'état
        if self.on_state_received:
            neuron_state = message.content.get('neuron_state', {})
            asyncio.create_task(self.on_state_received(message.sender_id, neuron_state))
            
    async def _handle_problem(self, message, connection):
        """Traite un message PROBLEM"""
        self.logger.debug(f"Received PROBLEM from {message.sender_id}")
        
        problem_id = message.content.get('problem_id')
        problem_type = message.content.get('problem_type')
        problem_data = message.content.get('problem_data')
        
        self.logger.info(f"New problem received: {problem_id} of type {problem_type}")
        
        # Si le hook est défini, transmettre le problème
        if self.on_problem_received:
            asyncio.create_task(self.on_problem_received(
                message.sender_id, problem_id, problem_type, problem_data
            ))
            
    async def _handle_solution(self, message, connection):
        """Traite un message SOLUTION"""
        self.logger.debug(f"Received SOLUTION from {message.sender_id}")
        
        solution_id = message.content.get('solution_id')
        problem_id = message.content.get('problem_id')
        solution_data = message.content.get('solution_data')
        
        self.logger.info(f"Solution received for problem {problem_id}")
        
        # Si le hook est défini, transmettre la solution
        if self.on_solution_received:
            asyncio.create_task(self.on_solution_received(
                message.sender_id, solution_id, problem_id, solution_data
            ))
            
    async def _handle_validation(self, message, connection):
        """Traite un message VALIDATION"""
        self.logger.debug(f"Received VALIDATION from {message.sender_id}")
        
        solution_id = message.content.get('solution_id')
        is_valid = message.content.get('is_valid')
        
        self.logger.info(f"Validation for solution {solution_id}: {'valid' if is_valid else 'invalid'}")
        
    async def _handle_knowledge(self, message, connection):
        """Traite un message KNOWLEDGE"""
        self.logger.debug(f"Received KNOWLEDGE from {message.sender_id}")
        
        knowledge_id = message.content.get('knowledge_id')
        knowledge_type = message.content.get('knowledge_type')
        
        self.logger.info(f"New knowledge received: {knowledge_id} of type {knowledge_type}")
        
    async def _handle_query(self, message, connection):
        """Traite un message QUERY"""
        self.logger.debug(f"Received QUERY from {message.sender_id}")
        
        query_id = message.content.get('query_id')
        query_type = message.content.get('query_type')
        
        self.logger.info(f"Query received: {query_id} of type {query_type}")
        
        # Traiter selon le type de requête
        if query_type == "FIND_PEERS":
            await self._send_peer_list(connection)
            
    async def _handle_peer_list(self, message, connection):
        """Traite un message PEER_LIST"""
        self.logger.debug(f"Received PEER_LIST from {message.sender_id}")
        
        peers = message.content.get('peers', [])
        new_peers = 0
        
        # Ajouter les nouveaux pairs à notre liste
        for peer_data in peers:
            # Skip si c'est nous-même
            if peer_data.get('node_id') == self.node_id:
                continue
                
            # Skip si blacklisté
            if peer_data.get('node_id') in self.blacklist:
                continue
                
            # Skip si déjà connu
            if peer_data.get('node_id') in self.peers:
                continue
                
            # Créer le nouveau PeerInfo
            try:
                peer_info = PeerInfo.from_dict(peer_data)
                self.peers[peer_info.node_id] = peer_info
                new_peers += 1
            except Exception as e:
                self.logger.error(f"Error adding peer: {e}")
                
        self.logger.info(f"Added {new_peers} new peers from peer list")
        
    async def _handle_ping(self, message, connection):
        """Traite un message PING"""
        self.logger.debug(f"Received PING from {message.sender_id}")
        
        # Répondre avec un PONG
        pong = Message.create_pong(self.node_id, message.msg_id)
        await connection.send_message(pong)
        
    async def _handle_pong(self, message, connection):
        """Traite un message PONG"""
        self.logger.debug(f"Received PONG from {message.sender_id}")
        
        # Vérifier si c'est une réponse à notre dernier ping
        if connection.last_ping_id and message.references and connection.last_ping_id in message.references:
            # Calculer la latence
            rtt = (time.time() - connection.last_ping_time) * 1000  # en ms
            connection.peer_info.update_latency(rtt)
            self.logger.debug(f"RTT to {message.sender_id}: {rtt:.2f} ms")
            
    async def _connect_to_bootstrap(self, host, port):
        """
        Se connecte à un nœud bootstrap
        
        Args:
            host (str): Adresse du nœud bootstrap
            port (int): Port du nœud bootstrap
            
        Returns:
            bool: True si connecté avec succès, False sinon
        """
        self.logger.info(f"Connecting to bootstrap node {host}:{port}")
        
        # Ajouter un délai aléatoire pour éviter les collisions lors des connexions simultanées
        await asyncio.sleep(random.uniform(0.1, 1.0))
        
        try:
            # Créer une socket et se connecter avec un timeout
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.setblocking(False)
            
            # Utiliser wait_for avec timeout pour éviter les blocages potentiels
            try:
                connect_coro = asyncio.get_event_loop().sock_connect(client_socket, (host, port))
                await asyncio.wait_for(connect_coro, timeout=10.0)  # Timeout de 10 secondes
            except (ConnectionRefusedError, OSError, asyncio.TimeoutError) as e:
                self.logger.error(f"Failed to connect to bootstrap node {host}:{port}: {e}")
                client_socket.close()
                return False
                
            # Créer un identifiant temporaire pour ce bootstrap
            temp_id = f"bootstrap_{uuid.uuid4().hex[:8]}"
            
            # Créer une entrée pour ce pair
            peer_info = PeerInfo(
                node_id=temp_id,
                address=host,
                port=port
            )
            
            # Établir la connexion
            connection = Connection(peer_info, client_socket)
            await connection.start()
            
            # Stocker temporairement avec vérification des duplications possibles
            if temp_id in self.connections:
                self.logger.warning(f"Overwriting existing connection with ID {temp_id}")
                await self.connections[temp_id].close()
                
            self.connections[temp_id] = connection
            
            # Envoyer un HELLO avec notre identité complète
            capabilities = {
                "type": "full_node",
                "version": "1.0.0",
                "port": self.port,  # Indiquer notre port d'écoute
                "features": ["quantum_sim", "neuron", "consensus"]
            }
            
            hello = Message.create_hello(
                sender_id=self.node_id,
                capabilities=capabilities,
                public_key="our_public_key"  # À remplacer par la vraie clé
            )
            
            # Utiliser wait_for pour l'envoi du message également
            try:
                send_coro = connection.send_message(hello)
                await asyncio.wait_for(send_coro, timeout=5.0)  # Timeout de 5 secondes
            except asyncio.TimeoutError:
                self.logger.error(f"Timeout sending HELLO to {host}:{port}")
                await connection.close()
                if temp_id in self.connections:
                    del self.connections[temp_id]
                return False
            
            # Lancer une tâche pour gérer cette connexion
            asyncio.create_task(self._handle_connection(connection))
            
            self.logger.info(f"Connected to bootstrap node {host}:{port}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error connecting to bootstrap node {host}:{port}: {e}")
            return False
            
    async def _network_maintenance(self):
        """Tâche de maintenance du réseau"""
        self.logger.info("Network maintenance task started")
        
        while self.running:
            try:
                # Maintenir un nombre minimum de connexions
                await self._ensure_minimum_connections()
                
                # Ping les connexions actives pour vérifier leur état
                await self._ping_active_connections()
                
                # Nettoyer les pairs inactifs depuis trop longtemps
                self._clean_inactive_peers()
                
                # Attendre avant la prochaine maintenance
                await asyncio.sleep(60)  # Toutes les minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in network maintenance: {e}")
                await asyncio.sleep(30)
                
        self.logger.info("Network maintenance task stopped")
        
    async def _ensure_minimum_connections(self):
        """S'assure d'avoir un nombre minimum de connexions actives"""
        active_connections = sum(1 for conn in self.connections.values() if conn.active)
        
        if active_connections >= self.min_connections:
            return
            
        # Calculer combien de nouvelles connexions établir
        to_connect = self.min_connections - active_connections
        
        # Trouver des pairs non connectés et non blacklistés
        available_peers = [
            peer_id for peer_id in self.peers
            if peer_id not in self.connections and peer_id not in self.blacklist
        ]
        
        # Si pas assez de pairs connus, rien à faire
        if not available_peers:
            self.logger.warning("Not enough known peers to maintain minimum connections")
            return
            
        # Mélanger pour diversifier les connexions
        random.shuffle(available_peers)
        
        # Se connecter à de nouveaux pairs
        connection_attempts = 0
        
        for peer_id in available_peers[:to_connect]:
            peer_info = self.peers[peer_id]
            
            # Skip si pair récemment vu comme inactif
            if not peer_info.is_active():
                continue
                
            try:
                # Créer une socket et se connecter
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.setblocking(False)
                
                try:
                    await asyncio.wait_for(
                        asyncio.get_event_loop().sock_connect(
                            client_socket, (peer_info.address, peer_info.port)
                        ),
                        timeout=5
                    )
                except (ConnectionRefusedError, OSError, asyncio.TimeoutError) as e:
                    self.logger.warning(f"Failed to connect to {peer_id} at {peer_info.address}:{peer_info.port}: {e}")
                    client_socket.close()
                    peer_info.connection_attempts += 1
                    continue
                    
                # Établir la connexion
                connection = Connection(peer_info, client_socket)
                await connection.start()
                
                # Stocker la connexion
                self.connections[peer_id] = connection
                peer_info.connection_active = True
                peer_info.connection_successes += 1
                
                # Envoyer un HELLO
                hello = Message.create_hello(
                    sender_id=self.node_id,
                    capabilities={"type": "full_node"},
                    public_key="our_public_key"  # À remplacer par la vraie clé
                )
                await connection.send_message(hello)
                
                # Lancer une tâche pour gérer cette connexion
                asyncio.create_task(self._handle_connection(connection))
                
                self.logger.info(f"Connected to peer {peer_id} at {peer_info.address}:{peer_info.port}")
                connection_attempts += 1
                
                # Mettre à jour la réputation
                peer_info.update_reputation(0.1)
                
            except Exception as e:
                self.logger.error(f"Error connecting to peer {peer_id}: {e}")
                
            # Limiter le nombre de tentatives par cycle
            if connection_attempts >= to_connect:
                break
                
        self.logger.info(f"Established {connection_attempts} new connections during maintenance")
        
    async def _ping_active_connections(self):
        """Envoie un ping à toutes les connexions actives"""
        for peer_id, connection in list(self.connections.items()):
            if not connection.active:
                continue
                
            try:
                await connection.ping()
            except Exception as e:
                self.logger.error(f"Error pinging peer {peer_id}: {e}")
                
    def _clean_inactive_peers(self):
        """Nettoie les pairs inactifs depuis trop longtemps"""
        # Supprimer les pairs non vus depuis plus de 24h
        inactive_cutoff = time.time() - 86400  # 24 heures
        
        inactive_peers = [
            peer_id for peer_id, peer in self.peers.items()
            if peer.last_seen < inactive_cutoff
        ]
        
        for peer_id in inactive_peers:
            del self.peers[peer_id]
            
        if inactive_peers:
            self.logger.info(f"Cleaned {len(inactive_peers)} inactive peers")
            
    async def _disconnect_peer(self, peer_id):
        """
        Déconnecte un pair
        
        Args:
            peer_id (str): Identifiant du pair à déconnecter
            
        Returns:
            bool: True si déconnecté avec succès, False sinon
        """
        if peer_id not in self.connections:
            return False
            
        connection = self.connections[peer_id]
        peer_info = connection.peer_info
        
        # Fermer la connexion
        await connection.close()
        
        # Mettre à jour l'état du pair
        if peer_id in self.peers:
            self.peers[peer_id].connection_active = False
            
        # Supprimer de la liste des connexions
        del self.connections[peer_id]
        
        self.logger.info(f"Disconnected from peer {peer_id}")
        
        # Appeler le hook si défini
        if self.on_peer_disconnected:
            asyncio.create_task(self.on_peer_disconnected(peer_id))
            
        return True
        
    async def _send_peer_list(self, connection):
        """
        Envoie la liste des pairs connus à une connexion
        
        Args:
            connection (Connection): Connexion destinataire
        """
        # Préparer la liste des pairs (limiter à 20 pour éviter message trop grand)
        peers_to_send = []
        
        # Prioriser les pairs actifs et avec bonne réputation
        sorted_peers = sorted(
            [p for p in self.peers.values() if p.node_id != connection.peer_info.node_id],
            key=lambda p: (p.is_active(), p.reputation),
            reverse=True
        )
        
        for peer in sorted_peers[:20]:
            peers_to_send.append(peer.to_dict())
            
        # Créer et envoyer le message
        peer_list_msg = Message.create_peer_list(
            sender_id=self.node_id,
            peers=peers_to_send
        )
        
        await connection.send_message(peer_list_msg)
        
    async def broadcast_state(self, neuron_state):
        """
        Diffuse l'état du neurone à tous les pairs connectés
        
        Args:
            neuron_state (dict): État du neurone à diffuser
            
        Returns:
            int: Nombre de pairs auxquels l'état a été envoyé
        """
        # Nettoyer l'état pour transmission (éviter les grands tableaux)
        clean_state = neuron_state.copy()
        if 'space_time_data' in clean_state:
            del clean_state['space_time_data']
            
        # Créer le message
        state_msg = Message.create_state(
            sender_id=self.node_id,
            neuron_state=clean_state
        )
        
        # Diffuser à tous les pairs connectés
        sent_count = 0
        
        for connection in self.connections.values():
            if connection.active:
                try:
                    await connection.send_message(state_msg)
                    sent_count += 1
                except Exception as e:
                    self.logger.error(f"Error sending state to {connection.peer_info.node_id}: {e}")
                    
        self.logger.info(f"Broadcasted neuron state to {sent_count} peers")
        return sent_count
        
    async def submit_problem(self, problem_data, problem_type):
        """
        Soumet un problème au réseau
        
        Args:
            problem_data (dict): Données du problème
            problem_type (str): Type de problème
            
        Returns:
            str: Identifiant du problème soumis
        """
        problem_id = str(uuid.uuid4())
        
        # Créer le message
        problem_msg = Message.create_problem(
            sender_id=self.node_id,
            problem_data=problem_data,
            problem_type=problem_type,
            problem_id=problem_id
        )
        
        # Diffuser à tous les pairs connectés
        sent_count = 0
        
        for connection in self.connections.values():
            if connection.active:
                try:
                    await connection.send_message(problem_msg)
                    sent_count += 1
                except Exception as e:
                    self.logger.error(f"Error sending problem to {connection.peer_info.node_id}: {e}")
                    
        self.logger.info(f"Submitted problem {problem_id} to {sent_count} peers")
        return problem_id
        
    async def submit_solution(self, problem_id, solution_data, confidence=None):
        """
        Soumet une solution à un problème
        
        Args:
            problem_id (str): Identifiant du problème
            solution_data (dict): Données de la solution
            confidence (float): Niveau de confiance dans la solution
            
        Returns:
            str: Identifiant de la solution soumise
        """
        # Créer le message
        solution_msg = Message.create_solution(
            sender_id=self.node_id,
            problem_id=problem_id,
            solution_data=solution_data,
            confidence=confidence
        )
        
        solution_id = solution_msg.content['solution_id']
        
        # Diffuser à tous les pairs connectés
        sent_count = 0
        
        for connection in self.connections.values():
            if connection.active:
                try:
                    await connection.send_message(solution_msg)
                    sent_count += 1
                except Exception as e:
                    self.logger.error(f"Error sending solution to {connection.peer_info.node_id}: {e}")
                    
        self.logger.info(f"Submitted solution {solution_id} for problem {problem_id} to {sent_count} peers")
        return solution_id
        
    async def share_knowledge(self, knowledge_data, knowledge_type, references=None):
        """
        Partage une connaissance avec le réseau
        
        Args:
            knowledge_data (dict): Données de la connaissance
            knowledge_type (str): Type de connaissance
            references (list): Références aux messages précédents
            
        Returns:
            str: Identifiant de la connaissance partagée
        """
        # Créer le message
        knowledge_msg = Message.create_knowledge(
            sender_id=self.node_id,
            knowledge_data=knowledge_data,
            knowledge_type=knowledge_type,
            references=references
        )
        
        knowledge_id = knowledge_msg.content['knowledge_id']
        
        # Diffuser à tous les pairs connectés
        sent_count = 0
        
        for connection in self.connections.values():
            if connection.active:
                try:
                    await connection.send_message(knowledge_msg)
                    sent_count += 1
                except Exception as e:
                    self.logger.error(f"Error sending knowledge to {connection.peer_info.node_id}: {e}")
                    
        self.logger.info(f"Shared knowledge {knowledge_id} with {sent_count} peers")
        return knowledge_id
        
    def get_network_stats(self):
        """
        Renvoie des statistiques sur l'état du réseau
        
        Returns:
            dict: Statistiques du réseau
        """
        active_connections = sum(1 for conn in self.connections.values() if conn.active)
        
        return {
            "node_id": self.node_id,
            "uptime": time.time() - self._start_time if hasattr(self, '_start_time') else 0,
            "known_peers": len(self.peers),
            "active_connections": active_connections,
            "blacklisted_peers": len(self.blacklist),
            "messages_sent": self._metrics.get("messages_sent", 0) if hasattr(self, '_metrics') else 0,
            "messages_received": self._metrics.get("messages_received", 0) if hasattr(self, '_metrics') else 0
        }