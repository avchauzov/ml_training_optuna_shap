import numpy as np

from tasks.metrics import calculate_metric


def calculate_error(y_true, y_pred, sample_weight=None, scoring=None):
	"""
	Calculate error based on the specified scoring method.

	Parameters:
	- prediction_list: Predicted values
	- y_list: Actual values
	- weight_list: Optional list of weights for each data point (default is None)
	- scoring: Scoring method (e.g., 'neg_mean_absolute_error', 'neg_mean_squared_error', 'sMAPE')

	Returns:
	- Calculated error using the specified scoring method
	"""
	if scoring is None:
		return np.nan  # Handle the case of undefined scoring
	
	if sample_weight is None:
		sample_weight = np.ones(len(y_true))  # Default weights are all ones
	
	return calculate_metric(scoring, y_true, y_pred, sample_weight=sample_weight)


def calculate_prediction_error(x_test, y_test, model, sample_weight=None, scoring=None):
	"""
	Calculate the error of predictions made by the model on test data.

	Parameters:
	- x_test: Test feature data
	- y_test: Test target data
	- model: Trained machine learning model
	- sample_weight: Optional list of weights for each test data point (default is None)
	- scoring: Scoring method (e.g., 'neg_mean_absolute_error', 'neg_mean_squared_error', 'sMAPE')

	Returns:
	- Calculated error of the model's predictions on the test data
	"""
	if scoring is None:
		return np.nan  # Handle the case of undefined scoring
	
	if sample_weight is None:
		sample_weight = np.ones(len(y_test))  # Default weights are all ones
	
	prediction_list = model.predict(x_test)
	
	return calculate_error(y_test, prediction_list, sample_weight, scoring)
