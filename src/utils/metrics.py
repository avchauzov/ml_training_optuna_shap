"""
This module contains a function to retrieve a dictionary of metrics and their properties for a given task.
"""

from src._settings.metrics import METRIC_FUNCTIONS
from src._settings.tasks import TASKS


def get_metric_dictionary_from_settings(task_name):
	"""
	Get a dictionary of metrics and their properties for a given task.

	Args:
		task_name (str): The name of the task.

	Returns:
		dict: A dictionary of metrics and their properties.
	"""
	# Retrieve the list of metric names associated with the given task
	metric_names = TASKS[task_name]
	
	# Create a dictionary containing metric names as keys and their properties from METRIC_FUNCTIONS
	metric_dictionary = {metric_name: METRIC_FUNCTIONS[metric_name] for metric_name in metric_names}
	
	return metric_dictionary
