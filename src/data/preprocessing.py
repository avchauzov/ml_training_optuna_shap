"""
This module contains functions for data preprocessing.
"""

from sklearn.preprocessing import MinMaxScaler

from src.data.loading import split_data


def preprocess_data_lightgbm(data, index, fillna=True):
	"""
	Preprocess data for LightGBM model.

	Args:
		data (list): List containing x_data, y_data, and weight_data.
		index (list): List containing train_index and test_index.
		fillna (bool, optional): Whether to fill NaN values. Default is True.

	Returns:
		tuple: A tuple containing x_train, y_train, weight_train, x_test, y_test, and weight_test.
	"""
	return filling_and_scaling_data(data, index, fillna=fillna)


def preprocess_data_multinomialnb(data, index, fillna=True):
	"""
	Preprocess data for MultinomialNB model.

	Args:
		data (list): List containing x_data, y_data, and weight_data.
		index (list): List containing train_index and test_index.
		fillna (bool, optional): Whether to fill NaN values. Default is True.

	Returns:
		tuple: A tuple containing x_train, y_train, weight_train, x_test, y_test, and weight_test.
	"""
	return filling_and_scaling_data(data, index, fillna=fillna)


def preprocess_data_sgdlinear(data, index, fillna=True, scale=True):
	"""
	Preprocess data for SGDLinear model.

	Args:
		data (list): List containing x_data, y_data, and weight_data.
		index (list): List containing train_index and test_index.
		fillna (bool, optional): Whether to fill NaN values. Default is True.
		scale (bool, optional): Whether to scale data. Default is True.

	Returns:
		tuple: A tuple containing x_train, y_train, weight_train, x_test, y_test, and weight_test.
	"""
	return filling_and_scaling_data(data, index, fillna=fillna, scale=scale)


def filling_and_scaling_data(data, index, fillna=True, scale=False):
	"""
	Fill NaN values and scale data.

	Args:
		data (list): List containing x_data, y_data, and weight_data.
		index (list): List containing train_index and test_index.
		fillna (bool, optional): Whether to fill NaN values. Default is True.
		scale (bool, optional): Whether to scale data. Default is False.

	Returns:
		tuple: A tuple containing x_train, y_train, weight_train, x_test, y_test, and weight_test.
	"""
	x_train, y_train, weight_train, x_test, y_test, weight_test = split_data(data, index)
	x_train, x_test = common_columns_selection(x_train, x_test)
	
	if fillna:
		fillna_df = x_train.mean()
		x_train = x_train.fillna(fillna_df).dropna(how='any', axis=1)
		x_test = x_test.fillna(fillna_df).dropna(how='any', axis=1)
	else:
		x_train = x_train.dropna(how='all', axis=1)
		x_test = x_test.dropna(how='all', axis=1)
	
	if scale:
		scaler = MinMaxScaler((0.1, 0.9))
		x_train = scaler.fit_transform(x_train)
		x_test = scaler.transform(x_test)
	
	return x_train, y_train, weight_train, x_test, y_test, weight_test


def common_columns_selection(x_train, x_test):
	"""
	Select common columns between training and test data.

	Args:
		x_train (DataFrame): Training data.
		x_test (DataFrame): Test data.

	Returns:
		tuple: A tuple containing x_train and x_test with common columns.
	"""
	common_columns_list = list(set(x_train.columns.tolist()) & set(x_test.columns.tolist()))
	x_train = x_train[common_columns_list]
	x_test = x_test[common_columns_list]
	return x_train, x_test


def preprocess_data(data, index, model_name):
	"""
	Preprocess data based on the specified model.

	Args:
		data (tuple): Tuple containing x_data, y_data, and weight_data.
		index (tuple): Tuple containing train_index and test_index.
		model_name (str): Name of the model.

	Returns:
		tuple: A tuple containing x_train, y_train, weight_train, x_test, y_test, and weight_test.
	"""
	x_data, y_data, weight_data = data
	train_index, test_index = index
	
	preprocess_functions = {
			'lightgbm'     : preprocess_data_lightgbm,
			'multinomialnb': preprocess_data_multinomialnb,
			'sgdlinear'    : preprocess_data_sgdlinear
			}
	
	preprocess_function = preprocess_functions[model_name]
	x_train, y_train, weight_train, x_test, y_test, weight_test = preprocess_function(
				[x_data, y_data, weight_data], [train_index, test_index]
				)
	
	return x_train, y_train, weight_train, x_test, y_test, weight_test
