import os
import sys

import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import KFold, StratifiedKFold

from automl.optimization import find_best_model


main_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, main_folder)


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


def run_optimization_for_test(task_name, model_name, metric_name, x_data, y_data, cv):
	best_hyperparameters_dictionary, important_features_list = find_best_model(
			task_name=task_name, model_name=model_name, metric_name=metric_name, x_data=x_data, y_data=y_data,
			weight_data=None, cv=cv, n_trials_long=32, n_trials_short=16, patience=8, n_jobs=16, test_mode=True
			)
	
	assert 1 <= len(important_features_list) <= 128


def test_lightgbm_classification_binary():
	x_data, y_data, cv = generate_data_and_split()
	run_optimization_for_test('classification_binary', 'lightgbm', 'roc_auc', x_data, y_data, cv)


def test_lightgbm_classification_multiclass():
	x_data, y_data, cv = generate_data_and_split(n_classes=3)
	run_optimization_for_test('classification_multiclass', 'lightgbm', 'roc_auc_ovr', x_data, y_data, cv)


def test_lightgbm_regression():
	x_data, y_data, cv = generate_data_and_split(classification=False)
	run_optimization_for_test('regression', 'lightgbm', 'neg_mean_squared_error', x_data, y_data, cv)


def test_sgdlinear_classification_binary():
	x_data, y_data, cv = generate_data_and_split()
	run_optimization_for_test('classification_binary', 'sgdlinear', 'roc_auc', x_data, y_data, cv)


def test_sgdlinear_classification_multiclass():
	x_data, y_data, cv = generate_data_and_split(n_classes=3)
	run_optimization_for_test('classification_multiclass', 'sgdlinear', 'roc_auc_ovr', x_data, y_data, cv)


def test_sgdlinear_regression():
	x_data, y_data, cv = generate_data_and_split(classification=False)
	run_optimization_for_test('regression', 'sgdlinear', 'neg_mean_squared_error', x_data, y_data, cv)


def test_multinomialnb_classification_multiclass():
	x_data, y_data, cv = generate_data_and_split(n_classes=3)
	run_optimization_for_test('classification_multiclass', 'multinomialnb', 'roc_auc_ovr', x_data, y_data, cv)


'''test_lightgbm_classification_binary()
test_lightgbm_classification_multiclass()
test_lightgbm_regression()
test_sgdlinear_classification_binary()
test_sgdlinear_classification_multiclass()
test_sgdlinear_regression()
test_multinomialnb_classification_multiclass()'''
