import numpy as np
from sklearn.preprocessing import OneHotEncoder

from automl.settings import METRICS


def calculate_test_error(data, model, metric_name, task_name):
	x_test, y_test, sample_weight_test = data
	
	if task_name in ['regression']:
		predictions = model.predict(x_test)
	
	else:
		predictions = model.predict_proba(x_test)
	
	return calculate_error([y_test, predictions, sample_weight_test], metric_name, task_name)


def calculate_error(data, metric_name, task_name):
	y_true, y_pred, sample_weight = data
	
	if task_name not in ['regression'] and len(y_true.shape) == 1:
		y_true = OneHotEncoder(sparse_output=False).fit_transform(np.array(y_true).reshape(-1, 1))
	
	return calculate_metric([y_true, y_pred, sample_weight], metric_name)


def calculate_metric(data, metric_name):
	y_true, y_pred, sample_weight = data
	metric_function = METRICS[metric_name][1]
	return metric_function(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
