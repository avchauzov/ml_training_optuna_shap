"""
This module defines a dictionary 'TASKS' that maps task names to corresponding metrics.
"""

# Dictionary that maps task names to corresponding metrics
TASKS = {
		'regression'               : [
				'neg_mean_absolute_error',
				'neg_mean_squared_error',
				'sMAPE'
				],
		'classification_binary'    : [
				'average_precision',
				'neg_log_loss',
				'neg_brier_score',
				'roc_auc'
				],
		'classification_multiclass': [
				'average_precision',
				'neg_log_loss',
				'roc_auc_ova',
				'roc_auc_ovr'
				]
		}
