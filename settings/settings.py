METRICS = {
		'neg_mean_absolute_error': 'minimize',
		'neg_mean_squared_error' : 'minimize',
		'sMAPE'                  : 'minimize',
		'average_precision'      : 'maximize',
		'neg_log_loss'           : 'minimize',
		'neg_brier_score'        : 'minimize',
		'roc_auc'                : 'maximize',
		'roc_auc_ova'            : 'maximize',
		'roc_auc_ovr'            : 'maximize'
		}

TASKS = {
		'regression'               : ['neg_mean_absolute_error', 'neg_mean_squared_error', 'sMAPE'],
		'classification_binary'    : ['average_precision', 'neg_log_loss', 'neg_brier_score', 'roc_auc'],
		'classification_multiclass': ['average_precision', 'neg_log_loss', 'roc_auc_ova', 'roc_auc_ovr']
		}

MODELS = {}
