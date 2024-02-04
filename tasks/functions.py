import numpy as np
from sklearn.metrics import (
	average_precision_score,
	brier_score_loss,
	log_loss,
	mean_absolute_error,
	mean_squared_error,
	roc_auc_score,
	)
from sklearn.preprocessing import OneHotEncoder

from utils.custom_metrics import symmetrical_mape


def calculate_test_error(data, model, scoring, problem_type):
	x_test, y_test, sample_weight_test = data
	
	if sample_weight_test is None:
		sample_weight_test = np.ones(len(y_test))
	
	if problem_type in ['regression']:
		predictions = model.predict(x_test)
	else:
		predictions = model.predict_proba(x_test)
	
	return calculate_error([y_test, predictions, sample_weight_test], scoring, problem_type)


def calculate_error(data, scoring, problem_type):
	y_true, y_pred, sample_weight = data
	
	if problem_type not in ['regression'] and len(y_true.shape) == 1:
		y_true = OneHotEncoder(sparse_output=False).fit_transform(y_true)
	
	return calculate_metric([y_true, y_pred, sample_weight], scoring)


def calculate_metric(data, scoring):
	metrics = {
			'neg_mean_absolute_error': mean_absolute_error,
			'neg_mean_squared_error' : mean_squared_error,
			'sMAPE'                  : symmetrical_mape,
			'average_precision'      : average_precision_score,
			'neg_brier_score'        : brier_score_loss,
			'neg_log_loss'           : log_loss,
			'roc_auc'                : roc_auc_score,
			'roc_auc_ova'            : lambda y_true, y_pred, sample_weight: roc_auc_score(y_true, y_pred, sample_weight=sample_weight, multi_class='ova'),
			'roc_auc_ovr'            : lambda y_true, y_pred, sample_weight: roc_auc_score(y_true, y_pred, sample_weight=sample_weight, multi_class='ovr'),
			}
	
	y_true, y_pred, sample_weight = data
	if scoring[0] in metrics:
		metric_function = metrics[scoring[0]]
		return metric_function(y_true, y_pred, sample_weight)
	
	raise ValueError(f'Error: Metric calculation function "{scoring[0]}" is not defined. Please add it to tasks -> functions.py -> calculate_metric(...) -> metrics variable!')
