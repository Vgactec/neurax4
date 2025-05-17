def process_grid(position):
    """
    Processes a grid position represented as a dictionary.

    Args:
        position (dict): A dictionary containing grid coordinates.

    Returns:
        list: A list of integer grid coordinates.
    """
    # Corrected dimensioning: Ensures that grid coordinates are integers.
    grid_position = [int(position[i]) for i in range(3)]
    return grid_position

# Example Usage:
position_data = {'0': '10', '1': '20', '2': '30'}
grid_coords = process_grid(position_data)
print(f"Grid Coordinates: {grid_coords}")