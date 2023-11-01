import numpy as np
from sklearn.utils import compute_sample_weight


def compute_sample_weights(x_train, y_train, x_test, y_test, weighing, scoring):
	"""
	Compute sample weights for training and test data based on specified conditions.

	Parameters:
	- y_train: True labels for training data
	- y_test: True labels for test data
	- weighing: Flag indicating whether to adjust weights
	- scoring: Scoring method (e.g., 'weighted')
	- x_train: Training feature data
	- x_test: Test feature data

	Returns:
	- Training and test data along with computed sample weights
	"""
	weight_train_list, weight_test_list = np.ones(len(y_train)), np.ones(len(y_test))
	
	if weighing:
		weight_train_list = compute_sample_weight('balanced', y_train)
		weight_test_list = compute_sample_weight('balanced', y_test)
	
	metric_weight_train_list, metric_weight_test_list = np.ones(len(y_train)), np.ones(len(y_test))
	
	if scoring[1] == 'weighted':
		metric_weight_train_list = compute_sample_weight('balanced', y_train)
		metric_weight_test_list = compute_sample_weight('balanced', y_test)
	
	return x_train, y_train, weight_train_list, metric_weight_train_list, x_test, y_test, weight_test_list, metric_weight_test_list
