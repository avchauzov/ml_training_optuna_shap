import numpy as np


def symmetrical_mape(y_list, prediction_list, sample_weight=None):
	"""
	Calculate the Symmetrical Mean Absolute Percentage Error (sMAPE).

	Parameters:
	- y_list: Actual values
	- prediction_list: Predicted values
	- sample_weight: Optional list of weights for each data point (default is None)

	Returns:
	- Symmetrical Mean Absolute Percentage Error (sMAPE)
	"""
	if sample_weight is None:
		sample_weight = np.ones(len(y_list))  # Default weights are all ones
	
	numerator = np.nansum(np.multiply(sample_weight, np.abs(y_list - prediction_list)))
	denominator = np.nansum(np.multiply(sample_weight, np.abs(y_list) + np.abs(prediction_list)))
	
	if denominator == 0:
		return 0.0  # Handle division by zero case
	
	return 1 / len(y_list) * numerator / denominator
