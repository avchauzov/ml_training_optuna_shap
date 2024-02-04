import numpy as np
from sklearn.preprocessing import OneHotEncoder

from tasks.metrics import calculate_metric


def calculate_error(y_true, y_pred, sample_weight=None, scoring=None):
	"""
	Calculate classification error based on the specified scoring method.

	Parameters:
	- prediction_list: Predicted probabilities or scores
	- y_list: True labels (multiclass labels should be one-hot encoded)
	- weight_list: Optional list of weights for each data point (default is None)
	- scoring: Scoring method (e.g., 'average_precision', 'neg_log_loss', 'roc_auc_ova', 'roc_auc_ovr')

	Returns:
	- Calculated classification error using the specified scoring method
	"""
	# Ensure y_list is one-hot encoded for appropriate metrics
	if len(y_true.shape) == 1:
		y_true = OneHotEncoder(sparse_output=False).fit_transform(y_true.reshape(-1, 1))
	
	if scoring is None:
		return np.nan  # Handle the case of undefined scoring
	
	if sample_weight is None:
		sample_weight = np.ones(len(y_true))  # Default weights are all ones
	
	return calculate_metric(scoring, y_true, y_pred, sample_weight=sample_weight)


def calculate_prediction_error(x_test, y_test, model, sample_weight=None, scoring=None):
	"""
	Calculate prediction error using a specified model and scoring method.

	Parameters:
	- x_test: Test feature data
	- y_test: True labels (multiclass labels should be one-hot encoded)
	- model: Trained machine learning model with a `predict_proba` method
	- sample_weight: Optional list of weights for test data points (default is None)
	- scoring: Scoring method (e.g., 'average_precision', 'neg_log_loss', 'roc_auc_ova', 'roc_auc_ovr')

	Returns:
	- Prediction error using the specified model and scoring method
	"""
	if scoring is None:
		return np.nan  # Handle the case of undefined scoring
	
	if sample_weight is None:
		sample_weight = np.ones(len(y_test))  # Default weights are all ones
	
	prediction_proba = model.predict_proba(x_test)
	
	return calculate_error(y_test, prediction_proba, sample_weight, scoring)
