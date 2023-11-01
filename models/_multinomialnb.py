from sklearn.naive_bayes import MultinomialNB

from data.preprocessing import preprocess_data_multinomialnb
from tasks.classification import compute_sample_weights as functions_classification_compute_sample_weights


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


def split_and_weight_data(x_data, y_data, train_index, test_index, weighing, scoring, task_name):
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
	x_train, y_train, x_test, y_test = preprocess_data_multinomialnb(x_data, y_data, train_index, test_index)
	
	if task_name == 'classification_multiclass':
		return functions_classification_compute_sample_weights(y_train, y_test, weighing, scoring, x_train, x_test)
	
	# Handle other task types or return None if not applicable
	return None
