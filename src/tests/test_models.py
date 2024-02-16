"""
This script contains functions for generating synthetic data and running hyperparameter optimization tests.
"""

import logging
import os
import random
import sys

import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import KFold, StratifiedKFold

from src.automl.optimization import find_best_model
from src.utils.tasks import TASKS


main_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, main_folder)

logging.basicConfig(level=logging.INFO)


def generate_data_and_split(classification=True, n_classes=2):
	"""
	Generate synthetic data and split it into training and test sets.

	Args:
		classification (bool): Whether to generate classification or regression data.
		n_classes (int): Number of classes for classification data.

	Returns:
		tuple: Tuple containing x_data, y_data, and cross-validation fold indices (cv).
	"""
	if classification:
		x_data, y_data = make_classification(n_samples=8096, n_features=128, n_informative=4, n_classes=n_classes)
	
	else:
		x_data, y_data = make_regression(n_samples=8096, n_features=128, n_informative=4)
	
	x_data = pd.DataFrame(x_data)
	
	cv = []
	if classification:
		kf = StratifiedKFold(n_splits=3, shuffle=True)
	else:
		kf = KFold(n_splits=3, shuffle=True)
	
	for train_index, test_index in kf.split(x_data, y_data):
		cv.append((train_index, test_index))
	
	return x_data, y_data, cv


def run_optimization_for_test(task_name, model_name, metric_name, x_data, y_data, cv):
	"""
	Run hyperparameter optimization for a specific test case.

	Args:
		task_name (str): Task name.
		model_name (str): Model name.
		metric_name (str): Metric name.
		x_data (pd.DataFrame): Input data.
		y_data (pd.Series): Target labels.
		cv (list): Cross-validation fold indices.

	Returns:
		None
	"""
	best_hyperparameters_dictionary, important_features_list = find_best_model(
			task_name=task_name, model_name=model_name, metric_name=metric_name, x_data=x_data, y_data=y_data,
			weight_data=None, cv=cv, n_trials_long=32, n_trials_short=16, patience=8, n_jobs=16, test_mode=True
			)


def test_lightgbm_classification_binary():
	"""
	Test hyperparameter optimization for LightGBM with binary classification.
	"""
	task_name = 'classification_binary'
	metric_name = random.choice(TASKS.get(task_name))
	
	x_data, y_data, cv = generate_data_and_split()
	run_optimization_for_test(task_name, 'lightgbm', metric_name, x_data, y_data, cv)


def test_lightgbm_classification_multiclass():
	"""
	Test hyperparameter optimization for LightGBM with multiclass classification.
	"""
	task_name = 'classification_multiclass'
	metric_name = random.choice(TASKS.get(task_name))
	n_classes = random.choice(list(range(3, 9)))
	
	x_data, y_data, cv = generate_data_and_split(n_classes=n_classes)
	run_optimization_for_test(task_name, 'lightgbm', metric_name, x_data, y_data, cv)


def test_lightgbm_regression():
	"""
	Test hyperparameter optimization for LightGBM with regression.
	"""
	task_name = 'regression'
	metric_name = random.choice(TASKS.get(task_name))
	
	x_data, y_data, cv = generate_data_and_split(classification=False)
	run_optimization_for_test(task_name, 'lightgbm', metric_name, x_data, y_data, cv)


def test_sgdlinear_classification_binary():
	"""
	Test hyperparameter optimization for SGD Linear with binary classification.
	"""
	task_name = 'classification_binary'
	metric_name = random.choice(TASKS.get(task_name))
	
	x_data, y_data, cv = generate_data_and_split()
	run_optimization_for_test(task_name, 'sgdlinear', metric_name, x_data, y_data, cv)


def test_sgdlinear_classification_multiclass():
	"""
	Test hyperparameter optimization for SGD Linear with multiclass classification.
	"""
	task_name = 'classification_multiclass'
	metric_name = random.choice(TASKS.get(task_name))
	n_classes = random.choice(list(range(3, 9)))
	
	x_data, y_data, cv = generate_data_and_split(n_classes=n_classes)
	run_optimization_for_test(task_name, 'sgdlinear', metric_name, x_data, y_data, cv)


def test_sgdlinear_regression():
	"""
	Test hyperparameter optimization for SGD Linear with regression.
	"""
	task_name = 'regression'
	metric_name = random.choice(TASKS.get(task_name))
	
	x_data, y_data, cv = generate_data_and_split(classification=False)
	run_optimization_for_test(task_name, 'sgdlinear', metric_name, x_data, y_data, cv)


def test_multinomialnb_classification_multiclass():
	"""
	Test hyperparameter optimization for Multinomial Naive Bayes with multiclass classification.
	"""
	task_name = 'classification_multiclass'
	metric_name = random.choice(TASKS.get(task_name))
	n_classes = random.choice(list(range(3, 9)))
	
	x_data, y_data, cv = generate_data_and_split(n_classes=n_classes)
	run_optimization_for_test(task_name, 'multinomialnb', metric_name, x_data.abs(), y_data, cv)


'''test_lightgbm_classification_binary()
test_lightgbm_classification_multiclass()
test_lightgbm_regression()
test_sgdlinear_classification_binary()
test_sgdlinear_classification_multiclass()
test_sgdlinear_regression()
test_multinomialnb_classification_multiclass()'''
