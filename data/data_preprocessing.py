from sklearn.preprocessing import MinMaxScaler

from data.data_loading import create_train_test_sets


def common_columns_selection(x_train, x_test):
	common_columns_list = list(set(x_train.columns.tolist()) & set(x_test.columns.tolist()))
	x_train = x_train[common_columns_list]
	x_test = x_test[common_columns_list]
	return x_train, x_test


def missing_values_removal(x_train, x_test, fillna):
	if fillna:
		fillna_df = x_train.mean()
		x_train = x_train.fillna(fillna_df).dropna(how='any', axis=1)
		x_test = x_test.fillna(fillna_df).dropna(how='any', axis=1)
	
	else:
		x_train = x_train.dropna(how='all', axis=1)
		x_test = x_test.dropna(how='all', axis=1)
	
	x_train, x_test = common_columns_selection(x_train, x_test)
	return x_train, x_test


def preprocess_data_lightgbm(data, index):
	x_train, y_train, weight_train, x_test, y_test, weight_test = create_train_test_sets(data, index)
	return x_train, y_train, weight_train, x_test, y_test, weight_test


def preprocess_data_multinomialnb(data, index):
	x_train, y_train, weight_train, x_test, y_test, weight_test = create_train_test_sets(data, index)
	x_train, x_test = missing_values_removal(x_train, x_test, True)
	
	return x_train, y_train, weight_train, x_test, y_test, weight_test


def preprocess_data_sgdlinear(data, index):
	x_train, y_train, weight_train, x_test, y_test, weight_test = create_train_test_sets(data, index)
	x_train, x_test = missing_values_removal(x_train, x_test, True)
	
	scaler = MinMaxScaler((0.1, 0.9))
	x_train = scaler.fit_transform(x_train)
	x_test = scaler.transform(x_test)
	
	return x_train, y_train, weight_train, x_test, y_test, weight_test
