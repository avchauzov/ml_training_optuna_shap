"""
This script defines a function to split data into training and test sets based on provided indices.
"""


def split_data(data, index):
	"""
	Split the data into training and test sets based on provided indices.

	Args:
		data (list): List containing x_data, y_data, and weight_data.
		index (list): List containing train_index and test_index.

	Returns:
		tuple: A tuple containing x_train, y_train, weight_train, x_test, y_test, and weight_test.
	"""
	x_data, y_data, weight_data = data
	train_index, test_index = index
	
	# Select training data
	x_train = x_data.iloc[train_index]
	y_train = y_data[train_index]
	weight_train = weight_data[train_index]
	
	# Select test data
	x_test = x_data.iloc[test_index]
	y_test = y_data[test_index]
	weight_test = weight_data[test_index]
	
	return x_train, y_train, weight_train, x_test, y_test, weight_test
