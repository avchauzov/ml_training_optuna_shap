from sklearn.linear_model import SGDClassifier, SGDRegressor

from data.preprocessing import preprocess_data_sgdlinear
from tasks.classification import compute_sample_weights as functions_classification_compute_sample_weights
from tasks.regression import compute_sample_weights as functions_regression_compute_sample_weights


def train_model(x_train, y_train, sample_weight, hyperparameters, task_name):
	"""
	Train a machine learning model based on the specified task and hyperparameters.

	Parameters:
	- x_train: Training feature data
	- y_train: Training target data
	- sample_weight: List of sample weights (optional)
	- hyperparameters: Dictionary of hyperparameters for the model
	- task_name: The type of task, can be 'regression', 'classification_binary', or 'classification_multiclass'

	Returns:
	- Trained machine learning model
	"""
	model = None
	
	if task_name == 'regression':
		model = SGDRegressor(**hyperparameters)
	elif task_name in ['classification_binary', 'classification_multiclass']:
		model = SGDClassifier(**hyperparameters)
	
	try:
		model.fit(x_train, y_train, sample_weight=sample_weight)
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
	- weighing: Weight adjustment parameter (if applicable)
	- scoring: Scoring parameter (if applicable)
	- task_name: The type of task, can be 'regression', 'classification_binary', or 'classification_multiclass'

	Returns:
	- Split and optionally weighted data
	"""
	# Preprocess the data
	x_train, y_train, x_test, y_test = preprocess_data_sgdlinear(x_data, y_data, train_index, test_index)
	
	if task_name == 'regression':
		return functions_regression_compute_sample_weights(x_train, y_train, x_test, y_test, weighing, scoring)
	elif task_name in ['classification_binary', 'classification_multiclass']:
		return functions_classification_compute_sample_weights(x_train, y_train, x_test, y_test, weighing, scoring)
	
	# Handle other task types or return None if not applicable
	return None
