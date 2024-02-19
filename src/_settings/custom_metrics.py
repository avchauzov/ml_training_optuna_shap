"""
This module contains a function to calculate Symmetrical Mean Absolute Percentage Error (sMAPE).
"""

import numpy as np


def calculate_symmetrical_mape(y_true, y_pred, sample_weight):
	"""
	Calculate Symmetrical Mean Absolute Percentage Error (sMAPE).

	Args:
		y_true (array-like): True target values.
		y_pred (array-like): Predicted values.
		sample_weight (array-like): Sample weights for each data point.

	Returns:
		float: Symmetrical Mean Absolute Percentage Error (sMAPE).
	"""
	# Calculate the absolute errors and apply sample weights
	absolute_errors = sample_weight * np.abs(y_true - y_pred)
	
	# Calculate the absolute sum of true and predicted values, applying sample weights
	# 1e-7 is added to avoid division by zero
	sum_absolute_values = sample_weight * (np.abs(y_true) + np.abs(y_pred)) + 1e-7
	
	# Calculate the sMAPE
	sMAPE = np.nanmean(2.0 * absolute_errors / sum_absolute_values)
	
	return sMAPE
