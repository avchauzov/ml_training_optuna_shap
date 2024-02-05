from data.data_preprocessing import preprocess_data_lightgbm, preprocess_data_multinomialnb, preprocess_data_sgdlinear


def split_and_weight_data(data, index, model_name):
	x_data, y_data, weight_data = data
	train_index, test_index = index
	
	if model_name in ['lightgbm']:
		x_train, y_train, weight_train, x_test, y_test, weight_test = preprocess_data_lightgbm([x_data, y_data, weight_data], [train_index, test_index])
	
	elif model_name in ['multinomialnb']:
		x_train, y_train, weight_train, x_test, y_test, weight_test = preprocess_data_multinomialnb([x_data, y_data, weight_data], [train_index, test_index])
	
	else:
		x_train, y_train, weight_train, x_test, y_test, weight_test = preprocess_data_sgdlinear([x_data, y_data, weight_data], [train_index, test_index])
	
	return x_train, y_train, weight_train, x_test, y_test, weight_test
