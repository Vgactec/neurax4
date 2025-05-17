import numpy as np
from scipy.constants import hbar, c, G
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

class QuantumGravitySimulator:
    def __init__(self, size=50, time_steps=10, grid_size=None):
        """
        Initialise le simulateur de gravité quantique
        
        Args:
            size (int): Taille de la grille 3D (utilisé si grid_size est None)
            time_steps (int): Nombre de pas temporels
            grid_size (int): Alternative à size, pour compatibilité avec les interfaces
        """
        # Pour assurer la compatibilité avec différentes interfaces
        if grid_size is not None:
            size = grid_size
            
        self.PLANCK_LENGTH = np.sqrt(hbar * G / c**3)
        self.PLANCK_TIME = self.PLANCK_LENGTH / c
        self.DEFAULT_INTENSITY = 1e-6
        
        self.size = size
        self.time_steps = time_steps
        self.space_time = np.zeros((time_steps, size, size, size))
        self.current_step = 0
        self.simulation_time = 0.0
        self.metrics = {}
        
        logging.info(f"Initialized simulator with grid size {size}³ and {time_steps} time steps")
        logging.info(f"Planck Length: {self.PLANCK_LENGTH}")
        logging.info(f"Initial space-time grid shape: {self.space_time.shape}")

    def quantum_fluctuations(self, intensity=2.0, time_slice=None):
        """
        Applique des fluctuations quantiques à l'espace-temps
        
        Args:
            intensity (float): Intensité des fluctuations (>0)
            time_slice (int, optional): Pas de temps spécifique à modifier (None = tous)
        
        Returns:
            ndarray: Référence à l'espace-temps modifié
        """
        # Vectorisation SIMD - Utilisation optimisée de numpy pour les calculs intensifs
        if time_slice is not None:
            # Appliquer les fluctuations à un pas de temps spécifique
            slice_shape = self.space_time[time_slice].shape
            noise = np.random.normal(0, intensity, slice_shape)
            exponential_factor = np.random.exponential(15.0, slice_shape)
            self.space_time[time_slice] += noise * exponential_factor  # Effets quantiques non-linéaires
        else:
            # Appliquer les fluctuations à tous les pas de temps
            noise = np.random.normal(0, intensity, self.space_time.shape)
            exponential_factor = np.random.exponential(15.0, self.space_time.shape)
            self.space_time += noise * exponential_factor  # Effets quantiques non-linéaires
        
        logging.debug(f"Applied quantum fluctuations with intensity {intensity}")
        return self.space_time

    def calculate_curvature(self, time_slice=None):
        """
        Calcule la courbure de l'espace-temps basée sur les fluctuations d'énergie-impulsion
        
        Args:
            time_slice (int, optional): Pas de temps spécifique à calculer (None = tous)
            
        Returns:
            ndarray: Courbure calculée
        """
        # Optimisation par vectorisation SIMD (Single Instruction, Multiple Data)
        if time_slice is not None:
            # Calculer la courbure pour un pas de temps spécifique
            data = self.space_time[time_slice]
            # Opérateur laplacien 3D (calculé en une seule opération vectorisée)
            laplacian = (
                np.roll(data, 1, axis=0) + 
                np.roll(data, -1, axis=0) +
                np.roll(data, 1, axis=1) + 
                np.roll(data, -1, axis=1) +
                np.roll(data, 1, axis=2) + 
                np.roll(data, -1, axis=2) - 
                6 * data
            )
            # Effet gravitationnel non-linéaire
            curvature = laplacian * self.PLANCK_LENGTH
            return curvature
        else:
            # Calculer la courbure pour tous les pas de temps
            curvatures = []
            for t in range(self.time_steps):
                curvatures.append(self.calculate_curvature(time_slice=t))
            return np.array(curvatures)

    def simulate_step(self, intensity=None):
        """
        Exécute une étape de simulation complète:
        1. Applique des fluctuations quantiques
        2. Calcule la courbure
        3. Met à jour l'espace-temps
        
        Args:
            intensity (float, optional): Intensité des fluctuations
            
        Returns:
            ndarray: État actuel de l'espace-temps après la simulation
        """
        if intensity is None:
            intensity = self.DEFAULT_INTENSITY
            
        t_idx = self.current_step % self.time_steps
        
        # Appliquer des fluctuations quantiques à l'étape de temps courante
        self.quantum_fluctuations(intensity, time_slice=t_idx)
        
        # Calculer la courbure résultante
        curvature = self.calculate_curvature(time_slice=t_idx)
        
        # Appliquer les effets gravitationnels à l'espace-temps
        self.space_time[t_idx] += curvature
        
        # Mettre à jour l'état de la simulation
        self.current_step += 1
        self.simulation_time += self.PLANCK_TIME
        
        # Mettre à jour les métriques
        self._update_metrics()
        
        return self.get_current_state()

    def simulate_multiple_steps(self, steps, intensity=None):
        """
        Exécute plusieurs étapes de simulation
        
        Args:
            steps (int): Nombre d'étapes à simuler
            intensity (float, optional): Intensité des fluctuations
            
        Returns:
            ndarray: État final de l'espace-temps
        """
        if intensity is None:
            intensity = self.DEFAULT_INTENSITY
            
        for _ in range(steps):
            self.simulate_step(intensity)
            
        return self.get_current_state()

    def _update_metrics(self):
        """
        Met à jour les métriques de la simulation
        """
        current_state = self.get_current_state()
        
        self.metrics = {
            'step': self.current_step,
            'simulation_time': self.simulation_time,
            'mean_value': float(np.mean(current_state)),
            'max_value': float(np.max(current_state)),
            'min_value': float(np.min(current_state)),
            'std_deviation': float(np.std(current_state)),
            'energy_density': float(np.sum(np.abs(current_state)) / (self.size**3))
        }

    def get_current_state(self):
        """
        Renvoie l'état actuel de l'espace-temps
        
        Returns:
            ndarray: Tranche actuelle de l'espace-temps
        """
        return self.space_time[self.current_step % self.time_steps].copy()

    def get_metrics(self):
        """
        Renvoie les métriques actuelles
        
        Returns:
            dict: Dictionnaire des métriques
        """
        return dict(self.metrics)

    def reset(self):
        """
        Réinitialise le simulateur à son état initial
        """
        self.space_time = np.zeros((self.time_steps, self.size, self.size, self.size))
        self.current_step = 0
        self.simulation_time = 0.0
        self.metrics = {}
        
        logging.info("Simulator reset to initial state")
        
    def set_grid_value(self, x, y, z, value, time_slice=0):
        """
        Définit une valeur spécifique dans la grille d'espace-temps
        
        Args:
            x (int): Coordonnée x
            y (int): Coordonnée y
            z (int): Coordonnée z
            value (float): Valeur à définir
            time_slice (int): Pas de temps (défaut: 0 = présent)
            
        Returns:
            bool: Succès de l'opération
        """
        if 0 <= x < self.size and 0 <= y < self.size and 0 <= z < self.size and 0 <= time_slice < self.time_steps:
            self.space_time[time_slice, x, y, z] = float(value)
            return True
        return False
        
    def get_grid_value(self, x, y, z, time_slice=0):
        """
        Récupère une valeur spécifique de la grille d'espace-temps
        
        Args:
            x (int): Coordonnée x
            y (int): Coordonnée y
            z (int): Coordonnée z
            time_slice (int): Pas de temps (défaut: 0 = présent)
            
        Returns:
            float: Valeur à la position indiquée
        """
        if 0 <= x < self.size and 0 <= y < self.size and 0 <= z < self.size and 0 <= time_slice < self.time_steps:
            return float(self.space_time[time_slice, x, y, z])
        return 0.0
    
    def extract_grid_slice(self, axes='xy', position=0, time_slice=0):
        """
        Extrait une tranche 2D de l'espace-temps
        Utile pour visualiser ou décoder des données comme les puzzles ARC
        
        Args:
            axes (str): Axes à extraire ('xy', 'xz', ou 'yz')
            position (int): Position sur le troisième axe
            time_slice (int): Pas de temps (défaut: 0 = présent)
            
        Returns:
            ndarray: Tranche 2D extraite
        """
        if 0 <= time_slice < self.time_steps:
            if axes == 'xy' and 0 <= position < self.size:
                return self.space_time[time_slice, :, :, position].copy()
            elif axes == 'xz' and 0 <= position < self.size:
                return self.space_time[time_slice, :, position, :].copy()
            elif axes == 'yz' and 0 <= position < self.size:
                return self.space_time[time_slice, position, :, :].copy()
        
        # En cas d'erreur, retourner une grille vide
        return np.zeros((self.size, self.size))