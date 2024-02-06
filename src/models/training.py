"""
This script provides functions for training machine learning models including LightGBM, Multinomial Naive Bayes,
and Stochastic Gradient Descent (SGD) Linear models.
"""

import lightgbm
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.naive_bayes import MultinomialNB


def train_lightgbm_model(data_train, data_test, hyperparameters, task_name):
	"""
	Train a LightGBM model based on the task and hyperparameters.

	Args:
		data_train (list): List containing x_train, y_train, and sample_weight_train.
		data_test (list): List containing x_test, y_test, and sample_weight_test.
		hyperparameters (dict): Hyperparameters for the LightGBM model.
		task_name (str): Task name.

	Returns:
		model: Trained LightGBM model.
	"""
	x_train, y_train, sample_weight_train = data_train
	x_test, y_test, sample_weight_test = data_test
	
	if task_name == 'regression':
		model = lightgbm.LGBMRegressor(**hyperparameters)
	else:
		model = lightgbm.LGBMClassifier(**hyperparameters)
	
	if hyperparameters['boosting_type'] == 'dart':
		model.fit(
				x_train, y_train,
				sample_weight=sample_weight_train,
				eval_set=[(x_test, y_test), (x_train, y_train)],
				eval_sample_weight=[sample_weight_test, sample_weight_train],
				eval_metric=hyperparameters['metric'],
				callbacks=[lightgbm.log_evaluation(period=-1, show_stdv=False)]
				)
	else:
		model.fit(
				x_train, y_train,
				sample_weight=sample_weight_train,
				eval_set=[(x_test, y_test), (x_train, y_train)],
				eval_sample_weight=[sample_weight_test, sample_weight_train],
				eval_metric=hyperparameters['metric'],
				callbacks=[
						lightgbm.early_stopping(int(hyperparameters['n_estimators'] * 0.10), verbose=False),
						lightgbm.log_evaluation(period=-1, show_stdv=False)
						]
				)
	
	return model


def train_multinomialnb_model(data, hyperparameters):
	"""
	Train a Multinomial Naive Bayes model.

	Args:
		data (list): List containing x_train, y_train, and sample_weight_train.
		hyperparameters (dict): Hyperparameters for the Multinomial Naive Bayes model.

	Returns:
		model: Trained Multinomial Naive Bayes model.
	"""
	x_train, y_train, sample_weight_train = data
	
	model = MultinomialNB(**hyperparameters)
	model.fit(x_train, y_train, sample_weight=sample_weight_train)
	return model


def train_sgdlinear_model(data, hyperparameters, task_name):
	"""
	Train a SGD Linear model based on the task and hyperparameters.

	Args:
		data (list): List containing x_train, y_train, and sample_weight_train.
		hyperparameters (dict): Hyperparameters for the SGD Linear model.
		task_name (str): Task name.

	Returns:
		model: Trained SGD Linear model.
	"""
	x_train, y_train, sample_weight_train = data
	
	if task_name == 'regression':
		model = SGDRegressor(**hyperparameters)
	else:
		model = SGDClassifier(**hyperparameters)
	
	model.fit(x_train, y_train, sample_weight=sample_weight_train)
	return model
