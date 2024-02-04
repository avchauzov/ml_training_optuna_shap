import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def create_train_test_sets(data, index):
	x_data, y_data, weight_data = data
	train_index, test_index = index
	
	x_train = pd.DataFrame(x_data.values[train_index])
	y_train = y_data[train_index]
	weight_train = weight_data[train_index]
	
	x_test = pd.DataFrame(x_data.values[test_index])
	y_test = y_data[test_index]
	weight_test = weight_data[test_index]
	
	return x_train, y_train, weight_train, x_test, y_test, weight_test


def preprocess_data_sgdlinear(data, index):
	"""
	Preprocess the input data for machine learning.

	Parameters:
	- x_data: Input feature data
	- y_data: Target data
	- train_index: Index of the training data
	- test_index: Index of the testing data

	Returns:
	- Preprocessed training and testing data
	"""
	x_train, y_train, weight_train, x_test, y_test, weight_test = create_train_test_sets(data, index)
	
	# Scale the features using Min-Max scaling
	scaler = MinMaxScaler((0.1, 0.9))
	x_train = scaler.fit_transform(x_train)
	x_test = scaler.transform(x_test)
	
	return x_train, y_train, weight_train, x_test, y_test, weight_test


def preprocess_data_multinomialnb(data, index):
	"""
	Preprocess the input data for machine learning.

	Parameters:
	- x_data: Input feature data
	- y_data: Target data
	- train_index: Index of the training data
	- test_index: Index of the testing data

	Returns:
	- Preprocessed training and testing data
	"""
	return create_train_test_sets(data, index)


def preprocess_data_lightgbm(data, index):
	"""
	Preprocess the input data for machine learning.

	Parameters:
	- x_data: Input feature data
	- y_data: Target data
	- train_index: Index of the training data
	- test_index: Index of the testing data

	Returns:
	- Preprocessed training and testing data
	"""
	x_train, y_train, weight_train, x_test, y_test, weight_test = create_train_test_sets(data, index)
	
	# Drop columns with all NaN values
	x_train = x_train.dropna(how='all', axis=1)
	x_test = x_test.dropna(how='all', axis=1)
	
	# Keep only common columns between train and test data
	common_columns_list = list(set(x_train.columns.tolist()) & set(x_test.columns.tolist()))
	x_train = x_train[common_columns_list]
	x_test = x_test[common_columns_list]
	
	return x_train, y_train, weight_train, x_test, y_test, weight_test


def compute_sample_weights(data_train, data_test, weighing, scoring):
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
	x_train, y_train, weight_train = data_train
	x_test, y_test, weight_test = data_test
	
	if scoring[1] == 'weighted':
		weight_metric_train, weight_metric_test = weight_train.copy(), weight_test.copy()
	
	else:
		weight_metric_train, weight_metric_test = np.ones(len(y_train)), np.ones(len(y_test))
	
	if not weighing:
		weight_train, weight_test = np.ones(len(y_train)), np.ones(len(y_test))
	
	return x_train, y_train, weight_train, weight_metric_train, x_test, y_test, weight_test, weight_metric_test


def missing_values_removal(data, index):
	x_train, y_train, weight_train, x_test, y_test, weight_test = create_train_test_sets(data, index)
	
	# Handle missing values by filling with mean
	fillna_df = x_train.mean()
	x_train = x_train.fillna(fillna_df).dropna(how='any', axis=1)
	x_test = x_test.fillna(fillna_df).dropna(how='any', axis=1)
	
	# Keep only common columns between train and test data
	common_columns_list = list(set(x_train.columns.tolist()) & set(x_test.columns.tolist()))
	x_train = x_train[common_columns_list]
	x_test = x_test[common_columns_list]
	
	return x_train, y_train, weight_train, x_test, y_test, weight_test
