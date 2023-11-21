from sklearn.naive_bayes import MultinomialNB

from data.preprocessing import compute_sample_weights, preprocess_data_multinomialnb


def train_model(x_train, y_train, sample_weight=None, hyperparameters=None):
	"""
	Train a Multinomial Naive Bayes model with optional sample weights.

	Parameters:
	- x_train: Training feature data
	- y_train: Training target data
	- sample_weight: Optional list of sample weights (default is None)
	- hyperparameters: Dictionary of hyperparameters for the model (default is None)

	Returns:
	- Trained Multinomial Naive Bayes model
	"""
	model = MultinomialNB(**(hyperparameters or {}))
	
	try:
		if sample_weight is not None:
			model.fit(x_train, y_train, sample_weight=sample_weight)
		else:
			model.fit(x_train, y_train)
		
		return model
	
	except Exception as e:
		print(f'ERROR: {e}')
		raise


def split_and_weight_data(x_data, y_data, weight_data, train_index, test_index, weight_adjustment, scoring, task_name):
	"""
	Split the data and perform weighting based on the specified task and parameters.

	Parameters:
	- x_data: Input feature data
	- y_data: Target data
	- train_index: Index of the training data
	- test_index: Index of the testing data
	- weighing: Flag indicating whether to adjust weights
	- scoring: Scoring method (e.g., 'weighted')
	- task_name: The type of task, can be 'classification_multiclass'

	Returns:
	- Split and optionally weighted data
	"""
	x_train, y_train, weight_train, x_test, y_test, weight_test = preprocess_data_multinomialnb(x_data, y_data, weight_data, train_index, test_index)
	return compute_sample_weights(x_train, y_train, weight_train, x_test, y_test, weight_test, weight_adjustment, scoring)
