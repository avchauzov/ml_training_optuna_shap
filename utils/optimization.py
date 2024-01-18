def get_metric_dictionary(task_name):
	"""
	Get a dictionary of scoring methods and their optimization directions based on the machine learning task.

	Parameters:
	- task_name: Machine learning task (e.g., 'regression', 'classification_binary', 'classification_multiclass')

	Returns:
	- Dictionary of scoring methods and their optimization directions
	"""
	if task_name == 'classification_binary':
		return {
				'average_precision': 'maximize',
				'neg_brier_score'  : 'minimize',
				'neg_log_loss'     : 'minimize',
				'roc_auc'          : 'maximize',
				}
	
	elif task_name == 'classification_multiclass':
		return {
				'average_precision': 'maximize',
				'neg_log_loss'     : 'minimize',
				'roc_auc_ova'      : 'maximize',
				'roc_auc_ovr'      : 'maximize',
				}
	
	elif task_name == 'regression':
		return {
				'neg_mean_absolute_error': 'minimize',
				'neg_mean_squared_error' : 'minimize',
				'sMAPE'                  : 'minimize',
				}


def generate_hyperparameter_space(task_name, model_name, _type, trial, scoring, num_class, n_jobs):
	"""
	Generates a hyperparameter search space based on the task, model, and other parameters.

	Args:
	    task_name (str): The machine learning task type (e.g., 'classification_binary', 'regression').
	    model_name (str): The model type (e.g., 'lightgbm', 'sgdlinear').
	    _type (str): The type of search space ('long' or 'short').
	    n_jobs (int): Number of CPU cores to use for training (e.g., 4).
	    trial (object): An instance of an optimization trial.
	    scoring (str): The scoring metric used for optimization (e.g., 'roc_auc_ovr', 'l2').
	    num_class (int): Number of classes (for multi-class tasks).

	Returns:
	    dict: A dictionary representing the hyperparameter search space.
	"""
	if task_name == 'classification_binary':
		if model_name == 'lightgbm':
			if _type == 'long':
				return {
						'num_leaves'       : trial.suggest_int('num_leaves', 2, 1024, step=1),
						'max_depth'        : trial.suggest_int('max_depth', 2, 16, step=1),
						'min_child_samples': trial.suggest_int('min_child_samples', 2, 1024, step=1),
						'max_bin'          : trial.suggest_int('max_bin', 16, 64, step=1),
						'min_data_in_bin'  : trial.suggest_int('min_data_in_bin', 2, 1024, step=1),
						'max_cat_threshold': trial.suggest_int('max_cat_threshold', 2, 128, step=1),
						'subsample_freq'   : trial.suggest_int('subsample_freq', 0, 1024, step=1),
						'n_estimators'     : trial.suggest_int('n_estimators', 128, 128, step=1),
						
						'extra_trees'      : trial.suggest_categorical('extra_trees', [True, False]),
						'boosting_type'    : trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'rf']),  # 'goss'
						'objective'        : trial.suggest_categorical('objective', ['binary']),  # 'cross_entropy'
						'metric'           : trial.suggest_categorical('metric', ['auc', 'average_precision', 'binary_logloss']),
						'weight_adjustment': trial.suggest_categorical('weight_adjustment', [True, False]),
						
						'learning_rate'    : trial.suggest_float('learning_rate', 0.1, 0.1, step=0.001),
						'reg_alpha'        : trial.suggest_float('reg_alpha', 1e-6, 128.0, log=True),
						'reg_lambda'       : trial.suggest_float('reg_lambda', 1e-6, 128.0, log=True),
						'min_split_gain'   : trial.suggest_float('min_split_gain', 1e-6, 128.0, log=True),
						'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.25, 1.0, step=0.05),
						'subsample'        : trial.suggest_float('subsample', 0.25, 0.95, step=0.05),
						'min_child_weight' : trial.suggest_float('min_child_weight', 1e-6, 1e6, log=True),
						'path_smooth'      : trial.suggest_float('path_smooth', 1e-6, 128.0, log=True),
						
						'n_jobs'           : trial.suggest_categorical('n_jobs', [n_jobs]),
						'verbosity'        : trial.suggest_categorical('verbosity', [-1]),
						'device'           : trial.suggest_categorical('device', ['gpu']),
						'gpu_use_dp'       : trial.suggest_categorical('gpu_use_dp', [False]),
						}
			else:
				return {
						'n_estimators' : trial.suggest_int('n_estimators', 64, 1024, step=1),
						'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, step=0.001),
						}
	
	if task_name == 'classification_multiclass':
		if model_name == 'lightgbm':
			objective_list = ['multiclass'] if scoring in ['roc_auc_ovr', 'roc_auc_ova'] else ['multiclass', 'multiclassova']
			if _type == 'long':
				return {
						'num_leaves'       : trial.suggest_int('num_leaves', 2, 1024),
						'max_depth'        : trial.suggest_int('max_depth', 2, 16),
						'min_child_samples': trial.suggest_int('min_child_samples', 2, 1024),
						'max_bin'          : trial.suggest_int('max_bin', 16, 64),
						'min_data_in_bin'  : trial.suggest_int('min_data_in_bin', 2, 1024),
						'max_cat_threshold': trial.suggest_int('max_cat_threshold', 2, 128),
						'subsample_freq'   : trial.suggest_int('subsample_freq', 0, 1024),
						'n_estimators'     : trial.suggest_int('n_estimators', 128, 128),
						
						'extra_trees'      : trial.suggest_categorical('extra_trees', [True, False]),
						'boosting_type'    : trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'rf']),  # 'goss'
						'objective'        : trial.suggest_categorical('objective', objective_list),
						'num_class'        : trial.suggest_categorical('num_class', [num_class]),
						'metric'           : trial.suggest_categorical('metric', ['auc_mu', 'multi_logloss', 'multi_error']),
						'weight_adjustment': trial.suggest_categorical('weight_adjustment', [True, False]),
						
						'learning_rate'    : trial.suggest_float('learning_rate', 0.1, 0.1, step=0.001),
						'reg_alpha'        : trial.suggest_float('reg_alpha', 1e-6, 128.0, log=True),
						'reg_lambda'       : trial.suggest_float('reg_lambda', 1e-6, 128.0, log=True),
						'min_split_gain'   : trial.suggest_float('min_split_gain', 1e-6, 128.0, log=True),
						'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.25, 1.0, step=0.05),
						'subsample'        : trial.suggest_float('subsample', 0.25, 0.95, step=0.05),
						'min_child_weight' : trial.suggest_float('min_child_weight', 1e-6, 1e6, log=True),
						'path_smooth'      : trial.suggest_float('path_smooth', 1e-6, 128.0, log=True),
						
						'n_jobs'           : trial.suggest_categorical('n_jobs', [n_jobs]),
						'verbosity'        : trial.suggest_categorical('verbosity', [-1]),
						'device'           : trial.suggest_categorical('device', ['gpu']),
						'gpu_use_dp'       : trial.suggest_categorical('gpu_use_dp', [False]),
						}
			else:
				return {
						'n_estimators' : trial.suggest_int('n_estimators', 64, 1024),
						'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, step=0.001),
						}
	
	if task_name == 'regression':
		if model_name == 'lightgbm':
			if _type == 'long':
				return {
						'num_leaves'       : trial.suggest_int('num_leaves', 2, 1024),
						'max_depth'        : trial.suggest_int('max_depth', 2, 16),
						'min_child_samples': trial.suggest_int('min_child_samples', 2, 1024),
						'max_bin'          : trial.suggest_int('max_bin', 16, 64),
						'min_data_in_bin'  : trial.suggest_int('min_data_in_bin', 2, 1024),
						'max_cat_threshold': trial.suggest_int('max_cat_threshold', 2, 128),
						'subsample_freq'   : trial.suggest_int('subsample_freq', 0, 1024),
						'n_estimators'     : trial.suggest_int('n_estimators', 128, 128),
						
						'extra_trees'      : trial.suggest_categorical('extra_trees', [True, False]),
						'boosting_type'    : trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'rf']),  # 'goss'
						'objective'        : trial.suggest_categorical(
								'objective', ['regression', 'regression_l1', 'huber', 'fair', 'quantile', 'mape']
								),  # 'poisson', 'gamma', 'tweedie'
						'metric'           : trial.suggest_categorical(
								'metric', ['l1', 'l2', 'rmse', 'quantile', 'mape', 'huber', 'fair', 'poisson', 'tweedie',
								           # 'gamma', 'gamma_deviance'
								           ]
								),
						'weight_adjustment': trial.suggest_categorical('weight_adjustment', [True, False]),
						
						'learning_rate'    : trial.suggest_float('learning_rate', 0.1, 0.1, step=0.001),
						'reg_alpha'        : trial.suggest_float('reg_alpha', 1e-6, 128.0, log=True),
						'reg_lambda'       : trial.suggest_float('reg_lambda', 1e-6, 128.0, log=True),
						'min_split_gain'   : trial.suggest_float('min_split_gain', 1e-6, 128.0, log=True),
						'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.25, 1.0, step=0.05),
						'subsample'        : trial.suggest_float('subsample', 0.25, 0.95, step=0.05),
						'min_child_weight' : trial.suggest_float('min_child_weight', 1e-6, 1e6, log=True),
						'path_smooth'      : trial.suggest_float('path_smooth', 1e-6, 128.0, log=True),
						
						'n_jobs'           : trial.suggest_categorical('n_jobs', [n_jobs]),
						'verbosity'        : trial.suggest_categorical('verbosity', [-1]),
						'device'           : trial.suggest_categorical('device', ['gpu']),
						'gpu_use_dp'       : trial.suggest_categorical('gpu_use_dp', [False]),
						}
			else:
				return {
						'n_estimators' : trial.suggest_int('n_estimators', 64, 1024),
						'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, step=0.001),
						}
	
	if task_name == 'classification_binary':
		if model_name == 'sgdlinear':
			if _type == 'long':
				return {
						'loss'             : trial.suggest_categorical('loss', ['log_loss', 'modified_huber']),
						'penalty'          : trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet']),
						'fit_intercept'    : trial.suggest_categorical('fit_intercept', [True, False]),
						'class_weight'     : trial.suggest_categorical('class_weight', [None, 'balanced']),
						'weight_adjustment': trial.suggest_categorical('weight_adjustment', [True, False]),
						
						'alpha'            : trial.suggest_float('alpha', 1e-6, 128.0, log=True),
						'l1_ratio'         : trial.suggest_float('l1_ratio', 0.0, 1.0, step=0.01),
						'epsilon'          : trial.suggest_float('epsilon', 1e-6, 128.0, log=True),
						
						'n_jobs'           : trial.suggest_categorical('n_jobs', [n_jobs]),
						'verbose'          : trial.suggest_categorical('verbose', [0]),
						}
			else:
				return {
						'max_iter'           : trial.suggest_int('max_iter', 512, 2048, step=1),
						'n_iter_no_change'   : trial.suggest_int('n_iter_no_change', 2, 16, step=1),
						'early_stopping'     : trial.suggest_categorical('early_stopping', [True]),
						'validation_fraction': trial.suggest_float('validation_fraction', 0.05, 0.25, step=0.05),
						}
	
	if task_name == 'classification_multiclass':
		if model_name == 'sgdlinear':
			if _type == 'long':
				return {
						'loss'             : trial.suggest_categorical('loss', ['modified_huber']),
						'penalty'          : trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet']),
						'fit_intercept'    : trial.suggest_categorical('fit_intercept', [True, False]),
						'class_weight'     : trial.suggest_categorical('class_weight', [None, 'balanced']),
						'weight_adjustment': trial.suggest_categorical('weight_adjustment', [True, False]),
						'early_stopping'   : trial.suggest_categorical('early_stopping', [False]),
						
						'alpha'            : trial.suggest_float('alpha', 1e-6, 128.0, log=True),
						'l1_ratio'         : trial.suggest_float('l1_ratio', 0.0, 1.0, step=0.01),
						'epsilon'          : trial.suggest_float('epsilon', 1e-6, 128.0, log=True),
						
						'n_jobs'           : trial.suggest_categorical('n_jobs', [n_jobs]),
						'verbose'          : trial.suggest_categorical('verbose', [0]),
						}
			else:
				return {
						'max_iter'      : trial.suggest_int('max_iter', 512, 2048, step=1),
						'early_stopping': trial.suggest_categorical('early_stopping', [False]),
						}
	
	if task_name == 'regression':
		if model_name == 'sgdlinear':
			if _type == 'long':
				return {
						'loss'             : trial.suggest_categorical('loss', ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']),
						'penalty'          : trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet']),
						'fit_intercept'    : trial.suggest_categorical('fit_intercept', [True, False]),
						'weight_adjustment': trial.suggest_categorical('weight_adjustment', [True, False]),
						
						'alpha'            : trial.suggest_float('alpha', 1e-6, 128.0, log=True),
						'l1_ratio'         : trial.suggest_float('l1_ratio', 0.0, 1.0, step=0.01),
						'epsilon'          : trial.suggest_float('epsilon', 1e-6, 128.0, log=True),
						
						'verbose'          : trial.suggest_categorical('verbose', [0]),
						}
			else:
				return {
						'max_iter'           : trial.suggest_int('max_iter', 512, 2048, step=1),
						'n_iter_no_change'   : trial.suggest_int('n_iter_no_change', 2, 16, step=1),
						'early_stopping'     : trial.suggest_categorical('early_stopping', [True]),
						'validation_fraction': trial.suggest_float('validation_fraction', 0.05, 0.25, step=0.05),
						}
	
	if task_name == 'classification_multiclass':
		if model_name == 'multinomialnb':
			return {
					'force_alpha'      : trial.suggest_categorical('force_alpha', [True, False]),
					'fit_prior'        : trial.suggest_categorical('fit_prior', [True, False]),
					'weight_adjustment': trial.suggest_categorical('weight_adjustment', [True, False]),
					
					'alpha'            : trial.suggest_float('alpha', 1e-6, 128.0, step=0.1),
					}
	
	return {}
