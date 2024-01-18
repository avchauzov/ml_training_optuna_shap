import os
import sys

import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import KFold, StratifiedKFold


main_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, main_folder)

from scripts.optimize import run_optimization


def generate_data_and_split(classification=True, n_classes=2):
	if classification:
		x_data, y_data = make_classification(n_samples=8096, n_features=128, n_informative=4, n_classes=n_classes)
		x_data = pd.DataFrame(x_data).abs()
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


def run_optimization_for_test(task_name, model_name, x_data, y_data, cv, scoring):
	best_hyperparameters_dictionary, weight_adjustment, important_features_list = run_optimization(
			task_name=task_name, model_name=model_name, x_data=x_data, y_data=y_data, weight_data=[1] * len(y_data),
			cv=cv, n_trials_long=32, n_trials_short=16, patience=8, scoring=scoring, calibration=None, n_jobs=16, test=True
			)
	
	assert weight_adjustment in [True, False]
	assert 1 <= len(important_features_list) <= 128


def test_lightgbm_classification_binary():
	x_data, y_data, cv = generate_data_and_split()
	run_optimization_for_test('classification_binary', 'lightgbm', x_data, y_data, cv, ('roc_auc', 'weighted'))  # add calibration


def test_lightgbm_classification_multiclass():
	x_data, y_data, cv = generate_data_and_split(n_classes=3)
	run_optimization_for_test('classification_multiclass', 'lightgbm', x_data, y_data, cv, ('roc_auc_ovr', 'weighted'))  # add calibration


def test_lightgbm_regression():
	x_data, y_data, cv = generate_data_and_split(classification=False)
	run_optimization_for_test('regression', 'lightgbm', x_data, y_data, cv, ('neg_mean_squared_error', 'weighted'))  # fix weighting


def test_sgdlinear_classification_binary():
	x_data, y_data, cv = generate_data_and_split()
	run_optimization_for_test('classification_binary', 'sgdlinear', x_data, y_data, cv, ('roc_auc', 'weighted'))  # fix weighting


def test_sgdlinear_classification_multiclass():
	x_data, y_data, cv = generate_data_and_split(n_classes=3)
	run_optimization_for_test('classification_multiclass', 'sgdlinear', x_data, y_data, cv, ('roc_auc_ovr', 'weighted'))  # fix weighting


def test_sgdlinear_regression():
	x_data, y_data, cv = generate_data_and_split(classification=False)
	run_optimization_for_test('regression', 'sgdlinear', x_data, y_data, cv, ('neg_mean_squared_error', 'weighted'))  # fix weighting


def test_multinomialnb_classification_multiclass():
	x_data, y_data, cv = generate_data_and_split(n_classes=3)
	run_optimization_for_test('classification_multiclass', 'multinomialnb', x_data, y_data, cv, ('roc_auc_ovr', 'weighted'))


'''test_lightgbm_classification_binary()
test_lightgbm_classification_multiclass()
test_lightgbm_regression()
test_sgdlinear_classification_binary()
test_sgdlinear_classification_multiclass()
test_sgdlinear_regression()
test_multinomialnb_classification_multiclass()'''
