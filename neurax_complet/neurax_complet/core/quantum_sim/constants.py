"""
Constantes physiques et paramètres de simulation pour le simulateur de gravité quantique
"""

from scipy import constants

# Constantes physiques fondamentales
SPEED_OF_LIGHT = constants.c       # Vitesse de la lumière (m/s)
GRAVITATIONAL_CONSTANT = constants.G  # Constante gravitationnelle (m^3 kg^-1 s^-2)
PLANCK_CONSTANT = constants.hbar   # Constante de Planck réduite (J·s)

# Longueur de Planck (m)
PLANCK_LENGTH = (PLANCK_CONSTANT * GRAVITATIONAL_CONSTANT / SPEED_OF_LIGHT**3)**0.5

# Temps de Planck (s)
PLANCK_TIME = PLANCK_LENGTH / SPEED_OF_LIGHT

# Paramètres de simulation
DEFAULT_GRID_SIZE = 64        # Taille de la grille par défaut
DEFAULT_TIME_STEPS = 8        # Nombre d'étapes temporelles par défaut
DEFAULT_INTENSITY = 1e-6      # Intensité des fluctuations quantiques par défaut

# Paramètres numériques
EPSILON = 1e-10               # Petit nombre pour éviter les divisions par zéro
MAX_FLOAT_VALUE = 1e30        # Valeur max pour limiter les instabilités numériques