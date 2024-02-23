"""
This module contains functions for calculating test errors and metrics, as well as updating dictionaries.
"""


def update_dict(space, hyperparameters):
	"""
	Update a dictionary with new keys and values from another dictionary.

	Args:
		space (dict): The dictionary to be updated.
		hyperparameters (dict): The dictionary containing new keys and values.

	Returns:
		dict: The updated dictionary.
	"""
	space.update((key, value) for key, value in hyperparameters.items() if key not in space)
	return space
