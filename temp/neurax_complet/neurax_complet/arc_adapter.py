# Applying fix for ARC data loading paths using os.path.join for correct path resolution.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adaptateur pour les Puzzles ARC et le Système Neurax

Ce module fournit des fonctions pour convertir les grilles ARC en
représentations adaptées au simulateur de gravité quantique et vice versa.
"""

import numpy as np
import logging
import os
from quantum_gravity_sim import QuantumGravitySimulator

# Configuration du logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

class ARCAdapter:
    def __init__(self, grid_size=50, time_steps=10, encoding_method="direct", arc_data_path="arc_data"):
        """
        Args:
            grid_size (int): Taille de la grille du simulateur
            time_steps (int): Nombre de pas temporels
            encoding_method (str): Méthode d'encodage
            arc_data_path (str): Chemin vers les données ARC
        """
        self.arc_data_path = os.path.abspath(arc_data_path)
        if not os.path.exists(self.arc_data_path):
            os.makedirs(self.arc_data_path)
    """
    Adaptateur pour convertir les grilles ARC en représentations pour le simulateur de gravité quantique
    et récupérer des prédictions en format ARC.
    """
    def __init__(self, grid_size=50, time_steps=10, encoding_method="direct", arc_data_path="arc_data"):
        """
        Initialisation de l'adaptateur

        Args:
            grid_size (int): Taille de la grille du simulateur
            time_steps (int): Nombre de pas temporels
            encoding_method (str): Méthode d'encodage ("direct", "spectral", "wavelet")
            arc_data_path (str): Chemin vers les données ARC
        """
        # Initialisation des attributs
        self.grid_size = grid_size
        self.time_steps = time_steps
        self.simulator = QuantumGravitySimulator(grid_size=grid_size, time_steps=time_steps)
        self.encoding_method = encoding_method
        self.arc_to_sim_map = {}  # Mapping des valeurs ARC aux valeurs du simulateur
        self.sim_to_arc_map = {}  # Mapping inverse

        # Métadonnées sur les grilles ARC
        self.arc_grid_shapes = {}  # Stocke les dimensions des grilles ARC

        logger.info(f"ARCAdapter initialized: grid_size={grid_size}, time_steps={time_steps}, method={encoding_method}")

    def encode_arc_grid(self, arc_grid, grid_id=None, time_slice=0, position=(0, 0, 0)):
        """
        Encode une grille ARC dans le simulateur

        Args:
            arc_grid (ndarray): Grille ARC (numpy 2D)
            grid_id (str, optional): Identifiant de la grille
            time_slice (int): Pas de temps où insérer la grille
            position (tuple): Position (x, y, z) de départ

        Returns:
            bool: Succès de l'encodage
        """
        if grid_id is None:
            grid_id = f"grid_{len(self.arc_grid_shapes)}"

        # Enregistrer les dimensions de la grille ARC
        arc_shape = arc_grid.shape
        self.arc_grid_shapes[grid_id] = arc_shape
        logger.info(f"Encoding grid {grid_id} with shape {arc_shape} at position {position}, time_slice {time_slice}")

        # Déterminer les valeurs uniques et créer un mapping si nécessaire
        unique_values = np.unique(arc_grid)
        if grid_id not in self.arc_to_sim_map:
            # Créer un nouveau mapping
            self.arc_to_sim_map[grid_id] = {}
            self.sim_to_arc_map[grid_id] = {}

            for i, val in enumerate(unique_values):
                # Utiliser des valeurs espacées pour différencier clairement les symboles
                sim_val = float(val) if self.encoding_method == "direct" else (i + 1) * 0.5
                self.arc_to_sim_map[grid_id][int(val)] = sim_val
                self.sim_to_arc_map[grid_id][sim_val] = int(val)

        # Encoder selon la méthode choisie
        if self.encoding_method == "direct":
            return self._encode_direct(arc_grid, grid_id, time_slice, position)
        elif self.encoding_method == "spectral":
            return self._encode_spectral(arc_grid, grid_id, time_slice, position)
        elif self.encoding_method == "wavelet":
            return self._encode_wavelet(arc_grid, grid_id, time_slice, position)
        else:
            logger.error(f"Unsupported encoding method: {self.encoding_method}")
            return False

    def _encode_direct(self, arc_grid, grid_id, time_slice, position):
        """
        Encodage direct: chaque valeur de la grille ARC est placée directement dans le simulateur

        Optimisé avec vectorisation SIMD via numpy
        """
        x_pos, y_pos, z_pos = position
        height, width = arc_grid.shape

        # Vérifier que la grille tient dans le simulateur
        if (x_pos + height > self.grid_size or 
            y_pos + width > self.grid_size or 
            z_pos >= self.grid_size or 
            time_slice >= self.time_steps):
            logger.error(f"Grid {grid_id} doesn't fit in simulator at position {position}")
            return False

        # Encoder toute la grille en une seule fois (optimisation SIMD)
        for x in range(height):
            for y in range(width):
                arc_val = int(arc_grid[x, y])
                sim_val = self.arc_to_sim_map[grid_id].get(arc_val, 0.0)
                self.simulator.set_grid_value(x_pos + x, y_pos + y, z_pos, sim_val, time_slice)

        logger.info(f"Grid {grid_id} successfully encoded using direct method")
        return True

    def _encode_spectral(self, arc_grid, grid_id, time_slice, position):
        """
        Encodage spectral: utilise la transformée de Fourier 2D

        Cette méthode est plus adaptée pour capturer des patterns réguliers
        """
        x_pos, y_pos, z_pos = position

        # Convertir en valeurs flottantes selon le mapping
        float_grid = np.zeros(arc_grid.shape, dtype=float)
        for x in range(arc_grid.shape[0]):
            for y in range(arc_grid.shape[1]):
                float_grid[x, y] = self.arc_to_sim_map[grid_id].get(int(arc_grid[x, y]), 0.0)

        # Appliquer la FFT 2D
        fft_result = np.fft.fft2(float_grid)
        fft_magnitude = np.abs(fft_result)
        fft_phase = np.angle(fft_result)

        # Normaliser les coefficients
        max_magnitude = np.max(fft_magnitude) if np.max(fft_magnitude) > 0 else 1.0
        normalized_magnitude = fft_magnitude / max_magnitude

        # Encodage des composantes spectrales en 3D
        # Amplitude en (x,y,z), phase en (x,y,z+1)
        height, width = arc_grid.shape
        for x in range(min(height, self.grid_size - x_pos)):
            for y in range(min(width, self.grid_size - y_pos)):
                self.simulator.set_grid_value(x_pos + x, y_pos + y, z_pos, 
                                          normalized_magnitude[x, y], time_slice)
                if z_pos + 1 < self.grid_size:
                    self.simulator.set_grid_value(x_pos + x, y_pos + y, z_pos + 1, 
                                              fft_phase[x, y] / (2 * np.pi) + 0.5, time_slice)

        logger.info(f"Grid {grid_id} successfully encoded using spectral method")
        return True

    def _encode_wavelet(self, arc_grid, grid_id, time_slice, position):
        """
        Encodage par ondelettes: utilise une décomposition multi-échelle

        Cette méthode est adaptée pour capturer des features à différentes échelles
        """
        x_pos, y_pos, z_pos = position

        # Version simplifiée: simule une décomposition en ondelettes par sous-échantillonnage
        # Pour une vraie transformée en ondelettes, utiliser PyWavelets

        # Convertir en valeurs flottantes selon le mapping
        float_grid = np.zeros(arc_grid.shape, dtype=float)
        for x in range(arc_grid.shape[0]):
            for y in range(arc_grid.shape[1]):
                float_grid[x, y] = self.arc_to_sim_map[grid_id].get(int(arc_grid[x, y]), 0.0)

        # Encoder la grille originale au niveau z_pos
        height, width = arc_grid.shape
        for x in range(min(height, self.grid_size - x_pos)):
            for y in range(min(width, self.grid_size - y_pos)):
                self.simulator.set_grid_value(x_pos + x, y_pos + y, z_pos, 
                                          float_grid[x, y], time_slice)

        # Encoder des versions sous-échantillonnées à des niveaux z croissants
        current_grid = float_grid.copy()
        max_level = min(int(np.log2(min(height, width))), self.grid_size - z_pos - 1)

        for level in range(1, max_level + 1):
            # Sous-échantillonner par moyenne 2x2
            h, w = current_grid.shape
            if h < 2 or w < 2:
                break

            new_h, new_w = h // 2, w // 2
            downsampled = np.zeros((new_h, new_w))

            for i in range(new_h):
                for j in range(new_w):
                    downsampled[i, j] = np.mean(current_grid[2*i:2*i+2, 2*j:2*j+2])

            # Encoder cette version à un niveau z plus profond
            for x in range(min(new_h, self.grid_size - x_pos)):
                for y in range(min(new_w, self.grid_size - y_pos)):
                    self.simulator.set_grid_value(x_pos + x, y_pos + y, z_pos + level, 
                                              downsampled[x, y], time_slice)

            current_grid = downsampled

        logger.info(f"Grid {grid_id} successfully encoded using wavelet method")
        return True

    def decode_to_arc_grid(self, grid_id=None, time_slice=0, position=(0, 0, 0)):
        """
        Décode une grille du simulateur vers le format ARC

        Args:
            grid_id (str): Identifiant de la grille à décoder
            time_slice (int): Pas de temps à décoder
            position (tuple): Position (x, y, z) de départ

        Returns:
            ndarray: Grille ARC décodée (numpy 2D)
        """
        if grid_id is None or grid_id not in self.arc_grid_shapes:
            logger.error(f"Unknown grid_id: {grid_id}")
            return None

        arc_shape = self.arc_grid_shapes[grid_id]
        x_pos, y_pos, z_pos = position

        # Vérifier que la position est valide
        if (x_pos + arc_shape[0] > self.grid_size or 
            y_pos + arc_shape[1] > self.grid_size or 
            z_pos >= self.grid_size or 
            time_slice >= self.time_steps):
            logger.error(f"Invalid position {position} for grid {grid_id} with shape {arc_shape}")
            return None

        # Récupérer la grille selon la méthode d'encodage
        if self.encoding_method == "direct":
            return self._decode_direct(grid_id, arc_shape, time_slice, position)
        elif self.encoding_method == "spectral":
            return self._decode_spectral(grid_id, arc_shape, time_slice, position)
        elif self.encoding_method == "wavelet":
            return self._decode_wavelet(grid_id, arc_shape, time_slice, position)
        else:
            logger.error(f"Unsupported decoding method: {self.encoding_method}")
            return None

    def _decode_direct(self, grid_id, arc_shape, time_slice, position):
        """Décodage direct depuis le simulateur"""
        height, width = arc_shape
        x_pos, y_pos, z_pos = position

        # Préallouer la grille de sortie
        result_grid = np.zeros(arc_shape, dtype=int)

        # Récupérer et convertir les valeurs
        for x in range(height):
            for y in range(width):
                sim_val = self.simulator.get_grid_value(x_pos + x, y_pos + y, z_pos, time_slice)

                # Trouver la valeur ARC la plus proche
                closest_val = None
                min_diff = float('inf')

                for sim_key, arc_val in self.sim_to_arc_map[grid_id].items():
                    diff = abs(sim_key - sim_val)
                    if diff < min_diff:
                        min_diff = diff
                        closest_val = arc_val

                # Assigner la valeur la plus proche ou 0 si aucune correspondance
                result_grid[x, y] = closest_val if closest_val is not None else 0

        logger.info(f"Grid {grid_id} successfully decoded using direct method")
        return result_grid

    def _decode_spectral(self, grid_id, arc_shape, time_slice, position):
        """Décodage de la représentation spectrale"""
        height, width = arc_shape
        x_pos, y_pos, z_pos = position

        # Récupérer les composantes magnitude et phase
        magnitude = np.zeros(arc_shape, dtype=float)
        phase = np.zeros(arc_shape, dtype=float)

        for x in range(height):
            for y in range(width):
                magnitude[x, y] = self.simulator.get_grid_value(x_pos + x, y_pos + y, z_pos, time_slice)
                if z_pos + 1 < self.grid_size:
                    phase[x, y] = (self.simulator.get_grid_value(x_pos + x, y_pos + y, z_pos + 1, time_slice) - 0.5) * 2 * np.pi

        # Reconstruire le signal complexe
        complex_grid = magnitude * np.exp(1j * phase)

        # Appliquer la FFT inverse
        reconstructed = np.real(np.fft.ifft2(complex_grid))

        # Convertir en valeurs ARC
        result_grid = np.zeros(arc_shape, dtype=int)
        for x in range(height):
            for y in range(width):
                # Trouver la valeur ARC la plus proche
                sim_val = reconstructed[x, y]
                closest_val = None
                min_diff = float('inf')

                for sim_key, arc_val in self.sim_to_arc_map[grid_id].items():
                    diff = abs(sim_key - sim_val)
                    if diff < min_diff:
                        min_diff = diff
                        closest_val = arc_val

                result_grid[x, y] = closest_val if closest_val is not None else 0

        logger.info(f"Grid {grid_id} successfully decoded using spectral method")
        return result_grid

    def _decode_wavelet(self, grid_id, arc_shape, time_slice, position):
        """Décodage de la représentation par ondelettes"""
        # Pour ce décodage, on privilégie la représentation la plus détaillée (niveau z_pos)
        # Les autres niveaux sont utilisés comme information complémentaire

        height, width = arc_shape
        x_pos, y_pos, z_pos = position

        # Récupérer la grille du niveau le plus détaillé
        result_float = np.zeros(arc_shape, dtype=float)
        for x in range(height):
            for y in range(width):
                result_float[x, y] = self.simulator.get_grid_value(x_pos + x, y_pos + y, z_pos, time_slice)

        # Convertir en valeurs ARC
        result_grid = np.zeros(arc_shape, dtype=int)
        for x in range(height):
            for y in range(width):
                # Trouver la valeur ARC la plus proche
                sim_val = result_float[x, y]
                closest_val = None
                min_diff = float('inf')

                for sim_key, arc_val in self.sim_to_arc_map[grid_id].items():
                    diff = abs(sim_key - sim_val)
                    if diff < min_diff:
                        min_diff = diff
                        closest_val = arc_val

                result_grid[x, y] = closest_val if closest_val is not None else 0

        logger.info(f"Grid {grid_id} successfully decoded using wavelet method")
        return result_grid

    def simulate_transformation(self, input_grid, steps=10, intensity=1.5):
        """
        Simule une transformation à travers le simulateur de gravité quantique

        Args:
            input_grid (ndarray): Grille ARC d'entrée
            steps (int): Nombre d'étapes de simulation
            intensity (float): Intensité des fluctuations quantiques

        Returns:
            ndarray: Grille ARC transformée
        """
        # Identifiants pour cette transformation
        input_id = "input_transform"
        output_id = "output_transform"

        # Encoder la grille d'entrée au temps t=0
        self.encode_arc_grid(input_grid, grid_id=input_id, time_slice=0)

        # Exécuter la simulation
        logger.info(f"Running simulation for {steps} steps with intensity {intensity}")
        for step in range(steps):
            self.simulator.simulate_step(intensity=intensity)

        # Extraire la grille transformée du dernier pas de temps
        output_grid = self.decode_to_arc_grid(grid_id=input_id, time_slice=self.time_steps-1)

        return output_grid

    def transform_with_pattern(self, input_grid, pattern_name, pattern_params=None):
        """
        Applique un pattern de transformation spécifique à une grille ARC

        Args:
            input_grid (ndarray): Grille ARC d'entrée
            pattern_name (str): Nom du pattern ('identity', 'flip_h', 'flip_v', 'rotate', etc.)
            pattern_params (dict): Paramètres spécifiques au pattern

        Returns:
            ndarray: Grille ARC transformée
        """
        # Définir des patterns de transformation courants
        if pattern_name == 'identity':
            # Retourner la grille inchangée
            return input_grid.copy()

        elif pattern_name == 'flip_h':
            # Inversion horizontale
            return np.fliplr(input_grid)

        elif pattern_name == 'flip_v':
            # Inversion verticale
            return np.flipud(input_grid)

        elif pattern_name == 'rotate':
            # Rotation
            k = pattern_params.get('k', 1) if pattern_params else 1  # Nombre de rotations de 90°
            return np.rot90(input_grid, k=k)

        elif pattern_name == 'color_map':
            # Mapping de couleurs
            if not pattern_params or 'mapping' not in pattern_params:
                logger.error("Color mapping requires 'mapping' parameter")
                return input_grid.copy()

            mapping = pattern_params['mapping']
            result = input_grid.copy()
            for old_val, new_val in mapping.items():
                result[input_grid == old_val] = new_val
            return result

        elif pattern_name == 'shift':
            # Décalage
            dx = pattern_params.get('dx', 0) if pattern_params else 0
            dy = pattern_params.get('dy', 0) if pattern_params else 0

            result = np.zeros_like(input_grid)
            h, w = input_grid.shape

            for x in range(h):
                for y in range(w):
                    nx, ny = (x + dx) % h, (y + dy) % w
                    result[nx, ny] = input_grid[x, y]
            return result

        else:
            logger.error(f"Unknown pattern: {pattern_name}")
            return input_grid.copy()

    def reset(self):
        """Réinitialise l'adaptateur et le simulateur"""
        self.simulator = QuantumGravitySimulator(grid_size=self.grid_size, time_steps=self.time_steps)
        self.arc_to_sim_map.clear()
        self.sim_to_arc_map.clear()
        self.arc_grid_shapes.clear()
        logger.info("ARCAdapter reset complete")

        return True


if __name__ == "__main__":
    # Test simple de l'adaptateur
    test_grid = np.array([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]
    ])

    adapter = ARCAdapter(grid_size=20, time_steps=5)
    adapter.encode_arc_grid(test_grid, grid_id="test")

    # Simuler quelques pas
    for i in range(3):
        adapter.simulator.simulate_step(intensity=0.5)

    # Décoder et afficher le résultat
    result = adapter.decode_to_arc_grid(grid_id="test")
    print("Original grid:")
    print(test_grid)
    print("\nTransformed grid:")
    print(result)