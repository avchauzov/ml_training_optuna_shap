"""
Script for training a LightGBM model and splitting/weighting data.

This script provides functions for training a LightGBM model based on the specified task and hyperparameters,
as well as for splitting the data and performing weighting based on the specified task and parameters.

Usage:
1. Use the 'train_model' function to train a LightGBM model.
2. Use the 'split_and_weight_data' function to split and optionally weight the data.
"""

import lightgbm

from data.preprocessing import compute_sample_weights, preprocess_data_lightgbm


def train_model(x_train, y_train, weight_train, x_test, y_test, weight_test, hyperparameters, task_name):
	"""
	Train a LightGBM model based on the specified task and hyperparameters.

	Parameters:
	- x_train: Training feature data
	- y_train: Training target data
	- weight_train: List of sample weights for training data
	- x_test: Test feature data
	- y_test: Test target data
	- weight_test: List of sample weights for test data
	- hyperparameters: Dictionary of hyperparameters for the model
	- task_name: The type of task, can be 'regression', 'classification_binary', or 'classification_multiclass'

	Returns:
	- Trained LightGBM model
	"""
	if task_name not in ['regression', 'classification_binary', 'classification_multiclass']:
		raise ValueError('Invalid task_name. Expected "regression", "classification_binary", or "classification_multiclass".')
	
	if task_name == 'regression':
		model = lightgbm.LGBMRegressor(**hyperparameters)
	else:
		model = lightgbm.LGBMClassifier(**hyperparameters)
	
	if hyperparameters['boosting_type'] != 'dart':
		model.fit(
				x_train, y_train,
				sample_weight=weight_train,
				eval_set=[(x_test, y_test), (x_train, y_train)],
				eval_sample_weight=[weight_test, weight_train],
				eval_metric=hyperparameters['metric'],
				callbacks=[
						lightgbm.early_stopping(int(hyperparameters['n_estimators'] * 0.10), verbose=False),
						lightgbm.log_evaluation(period=-1, show_stdv=False)
						]
				)
	else:
		model.fit(
				x_train, y_train,
				sample_weight=weight_train,
				eval_set=[(x_test, y_test), (x_train, y_train)],
				eval_sample_weight=[weight_test, weight_train],
				eval_metric=hyperparameters['metric'],
				callbacks=[lightgbm.log_evaluation(period=-1, show_stdv=False)]
				)
	
	return model


def split_and_weight_data(x_data, y_data, weight_data, train_index, test_index, weight_adjustment, scoring):
	"""
	Split the data and perform weighting based on the specified task and parameters.

	Parameters:
	- x_data: Input feature data
	- y_data: Target data
	- weight_data: Sample weights data
	- train_index: Index of the training data
	- test_index: Index of the testing data
	- weight_adjustment: Flag indicating whether to adjust weights
	- scoring: Scoring method (e.g., 'weighted')

	Returns:
	- Split and optionally weighted data
	"""
	x_train, y_train, weight_train, x_test, y_test, weight_test = preprocess_data_lightgbm(x_data, y_data, weight_data, train_index, test_index)
	return compute_sample_weights(x_train, y_train, weight_train, x_test, y_test, weight_test, weight_adjustment, scoring)
