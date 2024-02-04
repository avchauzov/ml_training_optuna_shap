from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score)

from utils.metrics import symmetrical_mape


def calculate_metric(scoring, y_true, y_pred, sample_weight=None):
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
	
	if scoring[0] in metrics:
		metric_function = metrics[scoring[0]]
		return metric_function(y_true, y_pred, sample_weight=sample_weight)
	
	raise ValueError(f'Error: Unknown metric: {scoring[0]}')
