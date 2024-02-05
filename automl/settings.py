from sklearn.metrics import (
	average_precision_score,
	brier_score_loss,
	log_loss,
	mean_absolute_error,
	mean_squared_error,
	roc_auc_score,
	)

from utils.custom_metrics import symmetrical_mape


TASKS = {
		'regression'               : ['neg_mean_absolute_error', 'neg_mean_squared_error', 'sMAPE'],
		'classification_binary'    : ['average_precision', 'neg_log_loss', 'neg_brier_score', 'roc_auc'],
		'classification_multiclass': ['average_precision', 'neg_log_loss', 'roc_auc_ova', 'roc_auc_ovr']
		}

MODELS = {'lightgbm', 'multinomialnb', 'sgdlinear'}

METRICS = {
		'neg_mean_absolute_error': ['minimize', mean_absolute_error],
		'neg_mean_squared_error' : ['minimize', mean_squared_error],
		'sMAPE'                  : ['minimize', symmetrical_mape],
		'neg_log_loss'           : ['minimize', log_loss],
		'average_precision'      : ['maximize', lambda y_true, y_pred, sample_weight: average_precision_score(y_true=y_true, y_score=y_pred, sample_weight=sample_weight)],
		'roc_auc'                : ['maximize', lambda y_true, y_pred, sample_weight: roc_auc_score(y_true=y_true, y_score=y_pred, sample_weight=sample_weight)],
		'neg_brier_score'        : ['minimize', lambda y_true, y_pred, sample_weight: brier_score_loss(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)],
		'roc_auc_ova'            : ['maximize', lambda y_true, y_pred, sample_weight: roc_auc_score(y_true=y_true, y_score=y_pred, sample_weight=sample_weight, multi_class='ova')],
		'roc_auc_ovr'            : ['maximize', lambda y_true, y_pred, sample_weight: roc_auc_score(y_true=y_true, y_score=y_pred, sample_weight=sample_weight, multi_class='ovr')]
		}
