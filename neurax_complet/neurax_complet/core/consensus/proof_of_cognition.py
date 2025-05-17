"""
Implémentation du mécanisme de consensus Preuve de Cognition (PoC)
Ce module définit les algorithmes de validation collective des résultats
"""

import numpy as np
import logging
import time
import hashlib
import json
import random
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class ValidationCriteria:
    """
    Critères de validation utilisés dans la Preuve de Cognition
    """
    # Critères généraux
    CONSISTENCY = "consistency"         # Cohérence physique/logique
    NOVELTY = "novelty"                 # Originalité/nouveauté
    UTILITY = "utility"                 # Utilité/applicabilité
    COMPLEXITY = "complexity"           # Complexité appropriée
    
    # Critères spécifiques aux solutions
    CORRECTNESS = "correctness"         # Exactitude mathématique
    COMPLETENESS = "completeness"       # Complétude de la solution
    EFFICIENCY = "efficiency"           # Efficacité/optimisation
    
    # Critères spécifiques aux connaissances
    COHERENCE = "coherence"             # Cohérence avec connaissances existantes
    GENERALIZABILITY = "generalizability"  # Potentiel de généralisation
    FALSIFIABILITY = "falsifiability"   # Possibilité de réfutation (scientifique)


class ValidationResult:
    """
    Résultat de validation d'une solution ou connaissance
    """
    
    def __init__(self, validation_id, item_id, validator_id, is_valid, 
                 criteria_scores=None, confidence=None, timestamp=None, 
                 explanation=None):
        """
        Initialise un résultat de validation
        
        Args:
            validation_id (str): Identifiant unique de cette validation
            item_id (str): Identifiant de l'élément validé
            validator_id (str): Identifiant du validateur
            is_valid (bool): True si l'élément est validé, False sinon
            criteria_scores (dict): Scores pour chaque critère
            confidence (float): Confiance dans la validation (0-1)
            timestamp (float): Horodatage de la validation
            explanation (str): Explication de la validation
        """
        self.validation_id = validation_id
        self.item_id = item_id
        self.validator_id = validator_id
        self.is_valid = is_valid
        self.criteria_scores = criteria_scores or {}
        self.confidence = confidence or 0.5
        self.timestamp = timestamp or time.time()
        self.explanation = explanation
        
    def get_weighted_score(self):
        """
        Calcule un score pondéré global basé sur tous les critères
        
        Returns:
            float: Score entre -1 (invalide avec certitude) et 1 (valide avec certitude)
        """
        if not self.criteria_scores:
            return 1.0 if self.is_valid else -1.0
            
        # Moyenne des scores de critères
        avg_score = sum(self.criteria_scores.values()) / len(self.criteria_scores)
        
        # Ajustement par la confiance et le résultat global
        direction = 1.0 if self.is_valid else -1.0
        return direction * avg_score * self.confidence
        
    def to_dict(self):
        """
        Convertit le résultat en dictionnaire
        
        Returns:
            dict: Représentation du résultat sous forme de dictionnaire
        """
        return {
            "validation_id": self.validation_id,
            "item_id": self.item_id,
            "validator_id": self.validator_id,
            "is_valid": self.is_valid,
            "criteria_scores": self.criteria_scores,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "explanation": self.explanation
        }
        
    @classmethod
    def from_dict(cls, data):
        """
        Crée un résultat à partir d'un dictionnaire
        
        Args:
            data (dict): Dictionnaire de données
            
        Returns:
            ValidationResult: Instance créée
        """
        return cls(
            validation_id=data.get("validation_id"),
            item_id=data.get("item_id"),
            validator_id=data.get("validator_id"),
            is_valid=data.get("is_valid"),
            criteria_scores=data.get("criteria_scores"),
            confidence=data.get("confidence"),
            timestamp=data.get("timestamp"),
            explanation=data.get("explanation")
        )


class ProofOfCognition:
    """
    Implémentation du mécanisme de consensus par Preuve de Cognition
    """
    
    def __init__(self, local_node_id, reputation_provider=None):
        """
        Initialise le mécanisme de consensus
        
        Args:
            local_node_id (str): Identifiant du nœud local
            reputation_provider: Fournisseur de scores de réputation pour les validateurs
        """
        self.local_node_id = local_node_id
        self.reputation_provider = reputation_provider
        
        # Stockage local des validations
        self.validations = {}  # {item_id: [ValidationResult, ...]}
        
        # Seuils de consensus
        self.min_validations = 3
        self.validation_timeout = 300  # secondes
        self.confidence_threshold = 0.7
        
        # Paramètres pour la sélection des validateurs
        self.default_num_validators = 5
        self.min_reputation = 0.3
        
    def select_validators(self, item_id, item_type, domain=None, num_validators=None):
        """
        Sélectionne les validateurs pour un élément
        
        Args:
            item_id (str): Identifiant de l'élément à valider
            item_type (str): Type d'élément (SOLUTION, KNOWLEDGE, etc.)
            domain (str): Domaine de l'élément (si applicable)
            num_validators (int): Nombre de validateurs à sélectionner
            
        Returns:
            list: Liste d'identifiants des validateurs sélectionnés
        """
        if self.reputation_provider is None:
            logger.warning("No reputation provider available, using random selection")
            return []
            
        num_validators = num_validators or self.default_num_validators
        
        # Obtenir la liste des pairs avec leur réputation
        peers = self.reputation_provider.get_peers_with_reputation()
        
        # Filtrer les pairs avec réputation minimum
        qualified_peers = [
            (peer_id, rep) for peer_id, rep in peers
            if rep >= self.min_reputation and peer_id != self.local_node_id
        ]
        
        # Si domaine spécifié, favoriser les pairs avec cette expertise
        if domain and hasattr(self.reputation_provider, 'get_domain_expertise'):
            domain_experts = self.reputation_provider.get_domain_expertise(domain)
            
            # Augmenter la réputation des experts du domaine
            enhanced_peers = []
            for peer_id, rep in qualified_peers:
                if peer_id in domain_experts:
                    # Bonus de réputation pour les experts du domaine
                    expertise_level = domain_experts[peer_id]
                    rep = min(1.0, rep * (1.0 + 0.5 * expertise_level))
                enhanced_peers.append((peer_id, rep))
            qualified_peers = enhanced_peers
            
        # Trier par réputation
        qualified_peers.sort(key=lambda x: x[1], reverse=True)
        
        # Nombre suffisant de pairs?
        if len(qualified_peers) < self.min_validations:
            logger.warning(f"Not enough qualified peers for validation: {len(qualified_peers)}/{self.min_validations}")
            # Prendre tous les pairs disponibles
            selected_peers = [peer_id for peer_id, _ in qualified_peers]
        else:
            # Sélection pondérée par réputation
            weights = [rep for _, rep in qualified_peers]
            total_weight = sum(weights)
            
            if total_weight <= 0:
                # Fallback à la sélection uniforme
                selected_indices = random.sample(range(len(qualified_peers)), 
                                               min(num_validators, len(qualified_peers)))
            else:
                # Normaliser les poids
                normalized_weights = [w/total_weight for w in weights]
                
                # Sélection avec remplacement pondérée
                selected_indices = np.random.choice(
                    len(qualified_peers),
                    size=min(num_validators, len(qualified_peers)),
                    replace=False,
                    p=normalized_weights
                )
                
            selected_peers = [qualified_peers[i][0] for i in selected_indices]
            
        logger.info(f"Selected {len(selected_peers)} validators for {item_type} {item_id}")
        return selected_peers
        
    def create_validation_request(self, item_id, item_type, item_data, validator_id):
        """
        Crée une requête de validation pour un validateur
        
        Args:
            item_id (str): Identifiant de l'élément à valider
            item_type (str): Type d'élément
            item_data (dict): Données de l'élément
            validator_id (str): Identifiant du validateur
            
        Returns:
            dict: Requête de validation
        """
        return {
            "request_id": hashlib.sha256(f"{item_id}:{validator_id}:{time.time()}".encode()).hexdigest()[:16],
            "item_id": item_id,
            "item_type": item_type,
            "item_data": item_data,
            "requester_id": self.local_node_id,
            "validator_id": validator_id,
            "timestamp": time.time(),
            "criteria": self._get_criteria_for_type(item_type)
        }
        
    def _get_criteria_for_type(self, item_type):
        """
        Renvoie les critères appropriés selon le type d'élément
        
        Args:
            item_type (str): Type d'élément
            
        Returns:
            list: Critères applicables
        """
        if item_type == "SOLUTION":
            return [
                ValidationCriteria.CONSISTENCY,
                ValidationCriteria.CORRECTNESS,
                ValidationCriteria.COMPLETENESS,
                ValidationCriteria.EFFICIENCY
            ]
        elif item_type == "KNOWLEDGE":
            return [
                ValidationCriteria.CONSISTENCY,
                ValidationCriteria.NOVELTY,
                ValidationCriteria.COHERENCE,
                ValidationCriteria.GENERALIZABILITY
            ]
        else:
            # Critères par défaut
            return [
                ValidationCriteria.CONSISTENCY,
                ValidationCriteria.UTILITY
            ]
            
    def add_validation_result(self, validation_result):
        """
        Ajoute un résultat de validation au stockage local
        
        Args:
            validation_result (ValidationResult): Résultat à ajouter
            
        Returns:
            bool: True si ajouté avec succès, False sinon
        """
        item_id = validation_result.item_id
        
        if item_id not in self.validations:
            self.validations[item_id] = []
            
        # Vérifier si ce validateur a déjà soumis un résultat
        for i, result in enumerate(self.validations[item_id]):
            if result.validator_id == validation_result.validator_id:
                # Remplacer l'ancien résultat
                self.validations[item_id][i] = validation_result
                logger.debug(f"Updated validation from {validation_result.validator_id} for {item_id}")
                return True
                
        # Ajouter le nouveau résultat
        self.validations[item_id].append(validation_result)
        logger.debug(f"Added validation from {validation_result.validator_id} for {item_id}")
        
        return True
        
    def process_validation_request(self, request, evaluation_function=None):
        """
        Traite une requête de validation entrante
        
        Args:
            request (dict): Requête de validation
            evaluation_function (callable): Fonction d'évaluation spécifique ou None
            
        Returns:
            ValidationResult: Résultat de la validation
        """
        item_id = request.get("item_id")
        item_type = request.get("item_type")
        item_data = request.get("item_data")
        requester_id = request.get("requester_id")
        criteria = request.get("criteria", [])
        
        logger.info(f"Processing validation request for {item_type} {item_id} from {requester_id}")
        
        # Si aucune fonction d'évaluation fournie, utiliser l'évaluation par défaut
        if evaluation_function is None:
            evaluation_function = self._default_evaluate
            
        # Effectuer l'évaluation
        try:
            criteria_scores, is_valid, confidence, explanation = evaluation_function(
                item_id, item_type, item_data, criteria
            )
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            # Valeurs par défaut en cas d'erreur
            criteria_scores = {c: 0.5 for c in criteria}
            is_valid = False
            confidence = 0.1
            explanation = f"Error during evaluation: {str(e)}"
            
        # Créer le résultat de validation
        result = ValidationResult(
            validation_id=hashlib.sha256(f"{item_id}:{self.local_node_id}:{time.time()}".encode()).hexdigest()[:16],
            item_id=item_id,
            validator_id=self.local_node_id,
            is_valid=is_valid,
            criteria_scores=criteria_scores,
            confidence=confidence,
            explanation=explanation
        )
        
        # Ajouter à notre stockage local
        self.add_validation_result(result)
        
        return result
        
    def _default_evaluate(self, item_id, item_type, item_data, criteria):
        """
        Évaluation par défaut quand aucune fonction spécifique n'est fournie
        
        Args:
            item_id (str): Identifiant de l'élément
            item_type (str): Type d'élément
            item_data (dict): Données de l'élément
            criteria (list): Critères à évaluer
            
        Returns:
            tuple: (criteria_scores, is_valid, confidence, explanation)
        """
        # Cette implémentation de base considère tous les éléments comme valides
        # avec une confiance modérée et des scores moyens
        # À remplacer par une vraie logique d'évaluation
        
        criteria_scores = {c: 0.7 for c in criteria}
        is_valid = True
        confidence = 0.6
        explanation = "Default evaluation with no specific validation logic."
        
        return criteria_scores, is_valid, confidence, explanation
        
    async def validate_item(self, item_id, item_type, item_data, domain=None, 
                          validator_ids=None, timeout=None, send_request_func=None):
        """
        Processus complet de validation d'un élément
        
        Args:
            item_id (str): Identifiant de l'élément à valider
            item_type (str): Type d'élément
            item_data (dict): Données de l'élément
            domain (str): Domaine de l'élément
            validator_ids (list): Liste d'identifiants des validateurs ou None
            timeout (float): Timeout en secondes
            send_request_func (callable): Fonction pour envoyer la requête aux validateurs
            
        Returns:
            tuple: (is_valid, confidence, explanations)
        """
        # Si aucune fonction d'envoi fournie, impossible de valider
        if send_request_func is None:
            logger.warning("No send_request_func provided, cannot validate externally")
            return False, 0.0, "No validation possible without send_request_func"
            
        # Déterminer les validateurs
        if validator_ids is None:
            validator_ids = self.select_validators(item_id, item_type, domain)
            
        if not validator_ids:
            logger.warning("No validators available")
            return False, 0.0, "No validators available"
            
        timeout = timeout or self.validation_timeout
        
        logger.info(f"Starting validation of {item_type} {item_id} with {len(validator_ids)} validators")
        
        # Créer et envoyer les requêtes
        validation_tasks = []
        
        for validator_id in validator_ids:
            # Créer la requête
            request = self.create_validation_request(item_id, item_type, item_data, validator_id)
            
            # Envoyer la requête de façon asynchrone
            task = asyncio.create_task(send_request_func(request, validator_id))
            validation_tasks.append(task)
            
        # Attendre les résultats avec timeout
        try:
            results = await asyncio.gather(*validation_tasks, return_exceptions=True)
        except asyncio.TimeoutError:
            logger.warning(f"Validation timed out for {item_id}")
            results = []
            
        # Traiter les résultats
        valid_results = []
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error during validation: {result}")
                continue
                
            if isinstance(result, ValidationResult):
                valid_results.append(result)
                # Ajouter à notre stockage local
                self.add_validation_result(result)
                
        # Ajouter notre propre validation si nous sommes aussi validateur
        if self.local_node_id not in validator_ids:
            # Auto-évaluation
            own_result = self.process_validation_request({
                "item_id": item_id,
                "item_type": item_type,
                "item_data": item_data,
                "requester_id": self.local_node_id,
                "criteria": self._get_criteria_for_type(item_type)
            })
            valid_results.append(own_result)
            
        # Calculer le consensus
        consensus = self.calculate_consensus(item_id)
        
        return consensus
        
    def calculate_consensus(self, item_id):
        """
        Calcule le consensus pour un élément spécifique
        
        Args:
            item_id (str): Identifiant de l'élément
            
        Returns:
            tuple: (is_valid, confidence, explanations)
        """
        if item_id not in self.validations or not self.validations[item_id]:
            return False, 0.0, "No validations available"
            
        validations = self.validations[item_id]
        
        # Pas assez de validations?
        if len(validations) < self.min_validations:
            logger.warning(f"Not enough validations for {item_id}: {len(validations)}/{self.min_validations}")
            
            # Si au moins une validation, utiliser le meilleur résultat
            if validations:
                # Trier par confiance
                validations.sort(key=lambda v: v.confidence, reverse=True)
                best = validations[0]
                return best.is_valid, best.confidence * 0.7, f"Preliminary result based on {len(validations)} validations"
            else:
                return False, 0.0, "No validations available"
                
        # Utiliser la réputation des validateurs comme poids si disponible
        weights = []
        
        if self.reputation_provider:
            for validation in validations:
                rep = self.reputation_provider.get_reputation(validation.validator_id) or 0.5
                weights.append(rep * validation.confidence)
        else:
            # Sinon, utiliser la confiance comme poids
            weights = [v.confidence for v in validations]
            
        # Normaliser les poids
        total_weight = sum(weights)
        if total_weight > 0:
            normalized_weights = [w/total_weight for w in weights]
        else:
            normalized_weights = [1.0/len(weights)] * len(weights)
            
        # Calculer le score pondéré
        weighted_score = sum(v.get_weighted_score() * w 
                            for v, w in zip(validations, normalized_weights))
        
        # Déterminer validité et confiance
        is_valid = weighted_score > 0
        confidence = abs(weighted_score)
        
        # Collecter les explications
        explanations = [f"{v.validator_id}: {v.explanation}" for v in validations if v.explanation]
        
        explanation = f"Consensus based on {len(validations)} validations with combined score {weighted_score:.2f}"
        if explanations:
            explanation += ". Details: " + "; ".join(explanations[:3])
            if len(explanations) > 3:
                explanation += f" and {len(explanations)-3} more."
                
        logger.info(f"Consensus for {item_id}: {'valid' if is_valid else 'invalid'} with confidence {confidence:.2f}")
        
        return is_valid, confidence, explanation
        
    def get_validator_weight(self, validator_id):
        """
        Calcule le poids d'un validateur
        
        Args:
            validator_id (str): Identifiant du validateur
            
        Returns:
            float: Poids entre 0 et 1
        """
        if not self.reputation_provider:
            return 0.5  # Poids moyen par défaut
            
        # Obtenir la réputation
        base_reputation = self.reputation_provider.get_reputation(validator_id) or 0.5
        
        # Ajustements supplémentaires pourraient être ajoutés ici
        
        return base_reputation
        
    def get_validation_history(self, item_id=None, validator_id=None):
        """
        Récupère l'historique des validations
        
        Args:
            item_id (str): Filtrer par ID d'élément
            validator_id (str): Filtrer par ID de validateur
            
        Returns:
            list: Liste des ValidationResults correspondants
        """
        results = []
        
        if item_id is not None:
            # Filtrer par élément spécifique
            if item_id in self.validations:
                if validator_id is not None:
                    # Filtrer aussi par validateur
                    results.extend([r for r in self.validations[item_id] 
                                  if r.validator_id == validator_id])
                else:
                    # Tous les validateurs pour cet élément
                    results.extend(self.validations[item_id])
        elif validator_id is not None:
            # Filtrer par validateur sur tous les éléments
            for item_validations in self.validations.values():
                results.extend([r for r in item_validations 
                              if r.validator_id == validator_id])
        else:
            # Toutes les validations
            for item_validations in self.validations.values():
                results.extend(item_validations)
                
        return results