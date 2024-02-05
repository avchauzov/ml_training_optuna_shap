import numpy as np

from data.data_weights import split_and_weight_data
from tasks.functions import calculate_test_error
from train.train_models import train_lightgbm_model, train_multinomialnb_model, train_sgdlinear_model


def cross_validate(data, cv, hyperparameters, task_name, model_name, metric_name):
	x_data, y_data, weight_data = data
	
	error = []
	for train_index, test_index in cv:
		
		if model_name in ['lightgbm']:
			x_train, y_train, weight_train, x_test, y_test, weight_test = split_and_weight_data([x_data, y_data, weight_data], [train_index, test_index], model_name)
			model = train_lightgbm_model([x_train, y_train, weight_train], [x_test, y_test, weight_test], hyperparameters, task_name)
		
		elif model_name in ['multinomialnb']:
			x_train, y_train, weight_train, x_test, y_test, weight_test = split_and_weight_data([x_data, y_data, weight_data], [train_index, test_index], model_name)
			model = train_multinomialnb_model([x_train, y_train, weight_train], hyperparameters)
		
		else:
			x_train, y_train, weight_train, x_test, y_test, weight_test = split_and_weight_data([x_data, y_data, weight_data], [train_index, test_index], model_name)
			model = train_sgdlinear_model([x_train, y_train, weight_train], hyperparameters, task_name)
		
		if task_name in ['classification_binary']:
			error.append(calculate_test_error([x_test, y_test, weight_test], model, metric_name, task_name))
		
		elif task_name in ['classification_multiclass']:
			error.append(calculate_test_error([x_test, y_test, weight_test], model, metric_name, task_name))
		
		else:
			error.append(calculate_test_error([x_test, y_test, weight_test], model, metric_name, task_name))
	
	return np.nanmean(error), np.nanstd(error)
