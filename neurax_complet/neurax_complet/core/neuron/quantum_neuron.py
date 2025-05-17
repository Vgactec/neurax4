#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module d'Implémentation du Neurone Quantique Gravitationnel

Ce module implémente les neurones quantiques capables d'apprendre à partir
des fluctuations de l'espace-temps simulées par le simulateur de gravité quantique.
Ces neurones servent de base à l'apprentissage des patterns abstraits.
"""

import numpy as np
import logging
import uuid
from scipy.special import expit  # Fonction sigmoïde
from ..quantum_sim.constants import *

logger = logging.getLogger(__name__)

class QuantumNeuron:
    """
    Implémentation d'un neurone quantique gravitationnel.

    Ce neurone utilise les propriétés quantiques de l'espace-temps pour apprendre
    et reconnaître des patterns complexes. Il combine des aspects de réseaux de neurones
    traditionnels avec des principes de mécanique quantique.
    """

    def __init__(self, input_dim=1, learning_rate=0.01, quantum_factor=0.5, use_bias=True, activation_type="lorentz"):
        """
        Initialise un neurone quantique.

        Args:
            input_dim (int): Dimension d'entrée du neurone
            learning_rate (float): Taux d'apprentissage initial
            quantum_factor (float): Facteur d'influence quantique (0-1)
            use_bias (bool): Utiliser un biais ou non
            activation_type (str): Type de fonction d'activation ('lorentz', 'sigmoid', 'tanh')
        """
        self.id = str(uuid.uuid4())[:8]  # Identifiant unique court
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.quantum_factor = quantum_factor
        self.use_bias = use_bias
        self.activation_type = activation_type

        # Initialisation des poids avec une distribution normale
        self.weights = np.random.normal(0, 0.1, input_dim)

        # Initialisation du biais
        self.bias = np.random.normal(0, 0.1) if use_bias else 0

        # Historique d'apprentissage
        self.training_history = []

        # Métriques quantiques
        self.quantum_state = np.zeros(input_dim)
        self.quantum_phase = 0.0
        self.coherence = 1.0  # Niveau de cohérence quantique (0-1)

        logger.info(f"Neurone quantique {self.id} initialisé: dim={input_dim}, qfactor={quantum_factor}")

    def _lorentz_activation(self, z, phi=None):
        """
        Fonction d'activation de Lorentz stabilisée: L(t) = 1 - e^{-t\phi(t)}
        """
        if phi is None:
            phi = min(abs(z), 100.0)  # Limite pour éviter l'overflow

        # Clip des valeurs pour stabilité numérique
        z_clipped = np.clip(z, -100.0, 100.0)
        product = -z_clipped * phi

        # Utiliser exp de manière sécurisée
        try:
            return 1 - np.exp(product)
        except:
            return 0.0 if product > 0 else 1.0

    def _sigmoid_activation(self, z):
        """Fonction d'activation sigmoïde: 1/(1+e^(-z))"""
        return expit(z)

    def _tanh_activation(self, z):
        """Fonction d'activation tangente hyperbolique: tanh(z)"""
        return np.tanh(z)

    def _quantum_modulation(self, input_values):
        """
        Applique une modulation quantique aux entrées.

        Cette fonction introduit des effets quantiques (superposition, interférence)
        dans le calcul du neurone.

        Args:
            input_values (numpy.ndarray): Valeurs d'entrée

        Returns:
            numpy.ndarray: Valeurs modulées
        """
        # Mise à jour de l'état quantique
        self.quantum_state = (1 - self.quantum_factor) * self.quantum_state + self.quantum_factor * input_values

        # Mise à jour de la phase quantique (rotation dans l'espace des phases)
        self.quantum_phase += 0.1 * np.sum(input_values) % (2 * np.pi)

        # Moduler les entrées avec des effets quantiques
        modulated_values = input_values + self.quantum_factor * (
            np.sin(self.quantum_phase) * self.quantum_state
        )

        # Appliquer un facteur de décohérence (réduction des effets quantiques avec le temps)
        self.coherence *= 0.99  # Décroissance lente de la cohérence
        self.coherence = max(0.1, self.coherence)  # Garder un minimum de cohérence

        return modulated_values

    def activate(self, input_values):
        """
        Calcule l'activation du neurone pour des valeurs d'entrée.

        Args:
            input_values (numpy.ndarray): Valeurs d'entrée (doit avoir la dimension input_dim)

        Returns:
            float: Valeur d'activation du neurone
        """
        # Vérifier la dimension d'entrée
        if np.isscalar(input_values):
            input_values = np.array([input_values])

        # Redimensionner si nécessaire
        if len(input_values) != self.input_dim:
            if len(input_values) > self.input_dim:
                input_values = input_values[:self.input_dim]
            else:
                # Padding avec des zéros
                padded = np.zeros(self.input_dim)
                padded[:len(input_values)] = input_values
                input_values = padded

        # Appliquer la modulation quantique
        modulated_input = self._quantum_modulation(input_values)

        # Calculer l'entrée pondérée
        z = np.dot(self.weights, modulated_input)
        if self.use_bias:
            z += self.bias

        # Appliquer la fonction d'activation appropriée
        if self.activation_type == "lorentz":
            return self._lorentz_activation(z)
        elif self.activation_type == "sigmoid":
            return self._sigmoid_activation(z)
        elif self.activation_type == "tanh":
            return self._tanh_activation(z)
        else:
            # Par défaut, utiliser Lorentz
            return self._lorentz_activation(z)

    def learn(self, input_values, target, learning_rate=None):
        """
        Ajuste les poids du neurone selon l'erreur observée.

        Implémente un apprentissage inspiré de la descente de gradient, mais
        avec des modifications quantiques pour améliorer l'exploration de l'espace des poids.

        Args:
            input_values (numpy.ndarray): Valeurs d'entrée
            target (float): Valeur cible attendue
            learning_rate (float, optional): Taux d'apprentissage spécifique pour cette itération

        Returns:
            float: Erreur après ajustement
        """
        # Utiliser le learning rate par défaut si non spécifié
        lr = learning_rate if learning_rate is not None else self.learning_rate

        # Calculer l'activation actuelle
        actual = self.activate(input_values)

        # Calculer l'erreur
        error = target - actual

        # Vérifier que input_values est un tableau
        if np.isscalar(input_values):
            input_values = np.array([input_values])

        # Appliquer la modulation quantique aux entrées
        modulated_input = self._quantum_modulation(input_values)

        # Ajuster les poids avec une composante quantique
        # La composante quantique ajoute une exploration stochastique guidée par l'état quantique
        # Cela permet de sortir des minimums locaux et d'explorer plus largement l'espace des solutions
        quantum_adjustment = self.quantum_factor * np.sin(self.quantum_phase) * self.quantum_state

        # Mise à jour des poids (combinaison d'apprentissage classique et quantique)
        weight_update = lr * error * modulated_input
        quantum_weight_update = lr * error * quantum_adjustment

        self.weights += weight_update + self.coherence * quantum_weight_update

        # Mise à jour du biais si utilisé
        if self.use_bias:
            self.bias += lr * error

        # Enregistrer l'historique d'apprentissage
        self.training_history.append({
            "error": error,
            "weights": self.weights.copy(),
            "bias": self.bias,
            "coherence": self.coherence
        })

        return error

    def batch_learn(self, input_batch, target_batch, epochs=1, learning_rate=None):
        """
        Apprentissage par lot sur plusieurs exemples.

        Args:
            input_batch (list): Liste de valeurs d'entrée
            target_batch (list): Liste de valeurs cibles
            epochs (int): Nombre d'époques d'apprentissage
            learning_rate (float, optional): Taux d'apprentissage

        Returns:
            dict: Résultats d'apprentissage avec erreurs moyennes par époque
        """
        results = {
            "epoch_errors": [],
            "final_weights": None,
            "final_bias": None,
            "convergence": False
        }

        # Validations des entrées
        if len(input_batch) != len(target_batch):
            raise ValueError("Le nombre d'entrées et de cibles doit être identique")

        # Apprentissage sur plusieurs époques
        for epoch in range(epochs):
            epoch_errors = []

            # Mélanger les données à chaque époque
            indices = np.random.permutation(len(input_batch))

            for i in indices:
                # Apprentissage sur chaque exemple
                error = self.learn(input_batch[i], target_batch[i], learning_rate)
                epoch_errors.append(error)

            # Calculer l'erreur moyenne de l'époque
            mean_error = np.mean(np.abs(epoch_errors))
            results["epoch_errors"].append(mean_error)

            # Vérifier la convergence (si l'erreur est suffisamment faible)
            if mean_error < 0.01:
                results["convergence"] = True
                break

        # Enregistrer les poids finaux
        results["final_weights"] = self.weights.copy()
        results["final_bias"] = self.bias

        return results

    def reset(self):
        """Réinitialise les poids et l'état du neurone"""
        self.weights = np.random.normal(0, 0.1, self.input_dim)
        self.bias = np.random.normal(0, 0.1) if self.use_bias else 0
        self.quantum_state = np.zeros(self.input_dim)
        self.quantum_phase = 0.0
        self.coherence = 1.0
        self.training_history = []

    def get_state(self):
        """
        Retourne l'état complet du neurone.

        Returns:
            dict: État du neurone
        """
        return {
            "id": self.id,
            "weights": self.weights.tolist(),
            "bias": float(self.bias),
            "input_dim": self.input_dim,
            "learning_rate": self.learning_rate,
            "quantum_factor": self.quantum_factor,
            "activation_type": self.activation_type,
            "quantum_state": self.quantum_state.tolist(),
            "quantum_phase": float(self.quantum_phase),
            "coherence": float(self.coherence)
        }

    def set_state(self, state):
        """
        Configure l'état du neurone à partir d'un dictionnaire.

        Args:
            state (dict): État du neurone
        """
        self.id = state.get("id", self.id)
        self.weights = np.array(state.get("weights", self.weights))
        self.bias = state.get("bias", self.bias)
        self.input_dim = state.get("input_dim", self.input_dim)
        self.learning_rate = state.get("learning_rate", self.learning_rate)
        self.quantum_factor = state.get("quantum_factor", self.quantum_factor)
        self.activation_type = state.get("activation_type", self.activation_type)
        self.quantum_state = np.array(state.get("quantum_state", self.quantum_state))
        self.quantum_phase = state.get("quantum_phase", self.quantum_phase)
        self.coherence = state.get("coherence", self.coherence)


class NeuronalNetwork:
    """
    Réseau de neurones quantiques.

    Implémente un réseau de neurones utilisant les QuantumNeurons pour former
    une architecture d'apprentissage complète.
    """

    def __init__(self, layers_config, learning_rate=0.01, quantum_factor=0.5):
        """
        Initialise un réseau de neurones quantiques.

        Args:
            layers_config (list): Configuration des couches sous forme de liste d'entiers
                                 [input_dim, hidden1, hidden2, ..., output_dim]
            learning_rate (float): Taux d'apprentissage global
            quantum_factor (float): Facteur quantique global
        """
        self.id = str(uuid.uuid4())[:8]
        self.layers_config = layers_config
        self.learning_rate = learning_rate
        self.quantum_factor = quantum_factor

        # Création des couches de neurones
        self.layers = []

        # Couches cachées + sortie
        for i in range(1, len(layers_config)):
            layer = []
            input_dim = layers_config[i-1]

            # Créer les neurones de cette couche
            for _ in range(layers_config[i]):
                neuron = QuantumNeuron(
                    input_dim=input_dim,
                    learning_rate=learning_rate,
                    quantum_factor=quantum_factor
                )
                layer.append(neuron)

            self.layers.append(layer)

        # Stockage des activations pour la propagation avant
        self.activations = []

        logger.info(f"Réseau neuronal quantique {self.id} initialisé: config={layers_config}")

    def forward(self, input_values):
        """
        Propage l'entrée à travers le réseau (forward pass).

        Args:
            input_values (numpy.ndarray): Valeurs d'entrée

        Returns:
            numpy.ndarray: Valeurs de sortie du réseau
        """
        # Réinitialiser les activations
        self.activations = [input_values]

        # Couche par couche
        for layer in self.layers:
            layer_activations = []

            # Pour chaque neurone de la couche
            for neuron in layer:
                # Activation avec les sorties de la couche précédente
                activation = neuron.activate(self.activations[-1])
                layer_activations.append(activation)

            # Stocker les activations de cette couche
            self.activations.append(np.array(layer_activations))

        # Retourner les activations de la dernière couche
        return self.activations[-1]

    def backward(self, target_values, learning_rate=None):
        """
        Ajuste les poids du réseau selon l'erreur observée (backward pass).

        Cette implémentation est une version simplifiée de la rétropropagation,
        adaptée spécifiquement aux neurones quantiques.

        Args:
            target_values (numpy.ndarray): Valeurs cibles
            learning_rate (float, optional): Taux d'apprentissage spécifique

        Returns:
            float: Erreur moyenne
        """
        # Vérifier que la propagation avant a été effectuée
        if not self.activations:
            raise ValueError("La propagation avant (forward) doit être effectuée avant la rétropropagation")

        # Utiliser le learning rate par défaut si non spécifié
        lr = learning_rate if learning_rate is not None else self.learning_rate

        # Calculer l'erreur de sortie
        output_values = self.activations[-1]
        output_errors = target_values - output_values

        # Pour chaque couche, de la dernière à la première
        for layer_idx in reversed(range(len(self.layers))):
            layer = self.layers[layer_idx]
            layer_input = self.activations[layer_idx]

            # Pour chaque neurone de la couche
            for neuron_idx, neuron in enumerate(layer):
                if layer_idx == len(self.layers) - 1:
                    # Dernière couche: utiliser l'erreur de sortie
                    error = output_errors[neuron_idx]
                else:
                    # Couches cachées: propager l'erreur
                    error = 0
                    next_layer = self.layers[layer_idx + 1]
                    for next_neuron in next_layer:
                        # Contribution à l'erreur basée sur les poids connectés
                        error += next_neuron.weights[neuron_idx] * next_neuron.training_history[-1]["error"]

                # Apprentissage du neurone
                neuron.learn(layer_input, 
                             neuron.activate(layer_input) + lr * error, 
                             learning_rate=lr)

        # Retourner l'erreur moyenne
        return np.mean(np.abs(output_errors))

    def train(self, input_batch, target_batch, epochs=100, learning_rate=None, validation_split=0.0, 
              early_stopping=False, patience=10, verbose=1):
        """
        Entraîne le réseau sur un ensemble de données.

        Args:
            input_batch (list): Liste des entrées d'entraînement
            target_batch (list): Liste des sorties cibles
            epochs (int): Nombre d'époques d'entraînement
            learning_rate (float, optional): Taux d'apprentissage
            validation_split (float): Portion des données à utiliser pour la validation (0.0-1.0)
            early_stopping (bool): Arrêter l'entraînement si pas d'amélioration
            patience (int): Nombre d'époques sans amélioration avant arrêt
            verbose (int): Niveau de détail des logs (0: aucun, 1: normal, 2: détaillé)

        Returns:
            dict: Historique d'entraînement (erreurs par époque)
        """
        if len(input_batch) != len(target_batch):
            raise ValueError("Le nombre d'entrées et de cibles doit être identique")

        # Convertir en tableaux numpy si nécessaire
        input_batch = np.array(input_batch)
        target_batch = np.array(target_batch)

        # Diviser en ensembles d'entraînement et de validation si demandé
        val_input = []
        val_target = []
        if validation_split > 0:
            val_size = int(len(input_batch) * validation_split)
            train_input = input_batch[:-val_size]
            train_target = target_batch[:-val_size]
            val_input = input_batch[-val_size:]
            val_target = target_batch[-val_size:]
        else:
            train_input = input_batch
            train_target = target_batch

        # Historique d'entraînement
        history = {
            "training_error": [],
            "validation_error": [] if validation_split > 0 else None,
            "best_epoch": 0,
            "convergence": False
        }

        # Variables pour early stopping
        best_val_error = float('inf')
        no_improvement_count = 0

        # Boucle d'entraînement
        for epoch in range(epochs):
            epoch_errors = []

            # Mélanger les données à chaque époque
            indices = np.random.permutation(len(train_input))

            # Pour chaque exemple d'entraînement
            for i in indices:
                # Propagation avant
                _ = self.forward(train_input[i])

                # Rétropropagation
                error = self.backward(train_target[i], learning_rate)
                epoch_errors.append(error)

            # Erreur moyenne d'entraînement
            mean_train_error = np.mean(epoch_errors)
            history["training_error"].append(mean_train_error)

            # Initialiser la variable mean_val_error avant utilisation
            mean_val_error = float('inf')

            # Validation si demandée
            if validation_split > 0 and len(val_input) > 0:
                val_errors = []
                for i in range(len(val_input)):
                    # Propagation avant uniquement (pas d'apprentissage)
                    pred = self.forward(val_input[i])
                    # Erreur
                    val_error = np.mean(np.abs(val_target[i] - pred))
                    val_errors.append(val_error)

                mean_val_error = np.mean(val_errors)
                history["validation_error"].append(mean_val_error)

                # Early stopping
                if early_stopping:
                    if mean_val_error < best_val_error:
                        best_val_error = mean_val_error
                        no_improvement_count = 0
                        history["best_epoch"] = epoch
                    else:
                        no_improvement_count += 1

                    if no_improvement_count >= patience:
                        if verbose > 0:
                            logger.info(f"Early stopping à l'époque {epoch+1}")
                        break

            # Afficher la progression
            if verbose > 0 and (epoch % max(1, epochs//10) == 0 or epoch == epochs-1):
                val_msg = f", val_error: {mean_val_error:.4f}" if validation_split > 0 and len(val_input) > 0 else ""
                logger.info(f"Époque {epoch+1}/{epochs}, error: {mean_train_error:.4f}{val_msg}")

            # Vérifier la convergence
            if mean_train_error < 0.01:
                history["convergence"] = True
                if verbose > 0:
                    logger.info(f"Convergence atteinte à l'époque {epoch+1}")
                break

        return history

    def predict(self, input_values):
        """
        Prédit la sortie pour une entrée donnée.

        Args:
            input_values: Valeurs d'entrée

        Returns:
            numpy.ndarray: Prédiction du réseau
        """
        return self.forward(input_values)

    def evaluate(self, input_batch, target_batch):
        """
        Évalue les performances du réseau sur un ensemble de test.

        Args:
            input_batch (list): Ensemble de test (entrées)
            target_batch (list): Ensemble de test (cibles)

        Returns:
            dict: Métriques de performance
        """
        if len(input_batch) != len(target_batch):
            raise ValueError("Le nombre d'entrées et de cibles doit être identique")

        errors = []
        predictions = []

        for i in range(len(input_batch)):
            pred = self.predict(input_batch[i])
            error = np.mean(np.abs(target_batch[i] - pred))
            errors.append(error)
            predictions.append(pred)

        return {
            "mean_error": np.mean(errors),
            "max_error": np.max(errors),
            "min_error": np.min(errors),
            "predictions": predictions
        }

    def save(self, file_path):
        """
        Sauvegarde le réseau dans un fichier.

        Args:
            file_path (str): Chemin du fichier de sauvegarde
        """
        import json

        # Construire la représentation du réseau
        network_data = {
            "id": self.id,
            "layers_config": self.layers_config,
            "learning_rate": self.learning_rate,
            "quantum_factor": self.quantum_factor,
            "layers": []
        }

        # Sauvegarder chaque couche
        for layer in self.layers:
            layer_data = []
            for neuron in layer:
                layer_data.append(neuron.get_state())
            network_data["layers"].append(layer_data)

        # Écrire dans le fichier
        with open(file_path, 'w') as f:
            json.dump(network_data, f, indent=2)

    @classmethod
    def load(cls, file_path):
        """
        Charge un réseau depuis un fichier.

        Args:
            file_path (str): Chemin du fichier de sauvegarde

        Returns:
            NeuronalNetwork: Réseau chargé
        """
        import json

        with open(file_path, 'r') as f:
            network_data = json.load(f)

        # Créer une nouvelle instance
        network = cls(
            layers_config=network_data["layers_config"],
            learning_rate=network_data["learning_rate"],
            quantum_factor=network_data["quantum_factor"]
        )

        network.id = network_data["id"]

        # Charger l'état de chaque neurone
        for layer_idx, layer_data in enumerate(network_data["layers"]):
            for neuron_idx, neuron_data in enumerate(layer_data):
                network.layers[layer_idx][neuron_idx].set_state(neuron_data)

        return network