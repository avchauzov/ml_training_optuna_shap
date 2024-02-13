"""
This script defines various evaluation metrics and associated functions for optimization tasks. It also provides a dictionary `METRIC_FUNCTIONS` that maps metric names to optimization directions and corresponding metric functions.
"""

from sklearn.metrics import (
	average_precision_score,
	brier_score_loss,
	log_loss,
	mean_absolute_error,
	mean_squared_error,
	roc_auc_score,
	)

from src.utils.custom_metrics import calculate_symmetrical_mape


# Dictionary mapping metric names to optimization directions and corresponding metric functions
METRIC_FUNCTIONS = {
		'neg_mean_absolute_error': ['minimize', mean_absolute_error],
		'neg_mean_squared_error' : ['minimize', mean_squared_error],
		'sMAPE'                  : ['minimize', calculate_symmetrical_mape],
		'neg_log_loss'           : ['minimize', log_loss],
		'average_precision'      : ['maximize', lambda y_true, y_pred, sample_weight: average_precision_score(y_true=y_true, y_score=y_pred, sample_weight=sample_weight)],
		'roc_auc'                : ['maximize', lambda y_true, y_pred, sample_weight: roc_auc_score(y_true=y_true, y_score=y_pred, sample_weight=sample_weight)],
		'neg_brier_score'        : ['minimize', lambda y_true, y_pred, sample_weight: brier_score_loss(y_true=y_true, y_prob=y_pred, sample_weight=sample_weight)],
		'roc_auc_ovo'            : ['maximize', lambda y_true, y_pred, sample_weight: roc_auc_score(y_true=y_true, y_score=y_pred, multi_class='ovo')],
		'roc_auc_ovr'            : ['maximize', lambda y_true, y_pred, sample_weight: roc_auc_score(y_true=y_true, y_score=y_pred, sample_weight=sample_weight, multi_class='ovr')]
		}
