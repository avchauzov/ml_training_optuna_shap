import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def preprocess_data_sgdlinear(x_data, y_data, train_index, test_index):
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
	x_train, y_train = pd.DataFrame(x_data.values[train_index]), y_data[train_index]
	x_test, y_test = pd.DataFrame(x_data.values[test_index]), y_data[test_index]
	
	# Handle missing values by filling with mean
	fillna_df = x_train.mean()
	x_train = x_train.fillna(fillna_df).dropna(how='any', axis=1)
	x_test = x_test.fillna(fillna_df).dropna(how='any', axis=1)
	
	# Keep only common columns between train and test data
	common_columns_list = list(set(x_train.columns.tolist()) & set(x_test.columns.tolist()))
	x_train = x_train[common_columns_list]
	x_test = x_test[common_columns_list]
	
	# Scale the features using Min-Max scaling
	scaler = MinMaxScaler((0.1, 0.9))
	x_train = scaler.fit_transform(x_train)
	x_test = scaler.transform(x_test)
	
	return x_train, y_train, x_test, y_test


def preprocess_data_multinomialnb(x_data, y_data, train_index, test_index):
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
	# Split the data into training and testing sets
	x_train, y_train = pd.DataFrame(x_data.values[train_index]), y_data[train_index]
	x_test, y_test = pd.DataFrame(x_data.values[test_index]), y_data[test_index]
	
	# Handle missing values by filling with mean
	fillna_df = x_train.mean()
	x_train = x_train.fillna(fillna_df).dropna(how='any', axis=1)
	x_test = x_test.fillna(fillna_df).dropna(how='any', axis=1)
	
	# Keep only common columns between train and test data
	common_columns_list = list(set(x_train.columns.tolist()) & set(x_test.columns.tolist()))
	x_train = x_train[common_columns_list]
	x_test = x_test[common_columns_list]
	
	return x_train, y_train, x_test, y_test


def preprocess_data_lightgbm(x_data, y_data, train_index, test_index):
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
	# Split the data into training and testing sets
	x_train, y_train = pd.DataFrame(x_data.values[train_index]), y_data[train_index]
	x_test, y_test = pd.DataFrame(x_data.values[test_index]), y_data[test_index]
	
	# Drop columns with all NaN values
	x_train = x_train.dropna(how='all', axis=1)
	x_test = x_test.dropna(how='all', axis=1)
	
	# Keep only common columns between train and test data
	common_columns_list = list(set(x_train.columns.tolist()) & set(x_test.columns.tolist()))
	x_train = x_train[common_columns_list]
	x_test = x_test[common_columns_list]
	
	return x_train, y_train, x_test, y_test
