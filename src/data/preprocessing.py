"""
This module contains functions for data preprocessing.
"""

from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler


def fill_and_scale_data(data, index, fillna=True):
	x_train, y_train, weight_train, x_test, y_test, weight_test = split_data_by_index(data, index)
	x_train, x_test = select_common_columns(x_train, x_test)
	
	if fillna:
		fillna_df = x_train.mean()
		x_train = x_train.fillna(fillna_df).dropna(how='any', axis=1)
		x_test = x_test.fillna(fillna_df).dropna(how='any', axis=1)
	else:
		x_train = x_train.dropna(how='all', axis=1)
		x_test = x_test.dropna(how='all', axis=1)
	
	return x_train, y_train, weight_train, x_test, y_test, weight_test


def scale_data(data, scaler_name):
	x_train, x_test = data
	
	scaler_functions = {
			'MaxAbsScaler'  : MaxAbsScaler(),
			'MinMaxScaler'  : MinMaxScaler(),
			'RobustScaler'  : RobustScaler(),
			'StandardScaler': StandardScaler()
			}
	
	scaler = scaler_functions.get(scaler_name)
	x_train = scaler.fit_transform(x_train)
	x_test = scaler.transform(x_test)
	
	return x_train, x_test


def preprocess_data(data, index, model_name):
	x_data, y_data, weight_data = data
	index_train, index_test = index
	
	preprocess_functions = {
			'lightgbm'          : preprocess_data_lightgbm,
			'multinomialnb'     : preprocess_data_multinomialnb,
			'sgdlinear'         : preprocess_data_linear,
			'elasticnet'        : preprocess_data_linear,
			'logisticregression': preprocess_data_linear
			}
	
	preprocess_function = preprocess_functions[model_name]
	return preprocess_function([x_data, y_data, weight_data], [index_train, index_test])


def preprocess_data_lightgbm(data, index, fillna=True):
	return fill_and_scale_data(data, index, fillna=fillna)


def preprocess_data_linear(data, index, fillna=True):
	return fill_and_scale_data(data, index, fillna=fillna)


def preprocess_data_multinomialnb(data, index, fillna=True):
	return fill_and_scale_data(data, index, fillna=fillna)


def select_common_columns(x_train, x_test):
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


def split_data_by_index(data, index):
	x_data, y_data, weight_data = data
	index_train, index_test = index
	
	# Select training data
	x_train = x_data.iloc[index_train]
	y_train = y_data[index_train]
	weight_train = weight_data[index_train]
	
	# Select test data
	x_test = x_data.iloc[index_test]
	y_test = y_data[index_test]
	weight_test = weight_data[index_test]
	
	return x_train, y_train, weight_train, x_test, y_test, weight_test
