import pandas as pd


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
