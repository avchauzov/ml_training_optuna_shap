def implement_lightgbm_production_mode(parameters, trial):
	parameters['device'] = trial.suggest_categorical('device', ['gpu'])
	parameters['gpu_use_dp'] = trial.suggest_categorical('gpu_use_dp', [False])
	
	return parameters


def lightgbm_long_parameters(trial, objective, metric, num_class, n_jobs):
	parameters = {
			'num_leaves'       : trial.suggest_int('num_leaves', 2, 1024),
			'max_depth'        : trial.suggest_int('max_depth', 2, 16),
			'min_child_samples': trial.suggest_int('min_child_samples', 2, 1024),
			'max_bin'          : trial.suggest_int('max_bin', 16, 64),
			'min_data_in_bin'  : trial.suggest_int('min_data_in_bin', 2, 1024),
			'max_cat_threshold': trial.suggest_int('max_cat_threshold', 2, 128),
			'subsample_freq'   : trial.suggest_int('subsample_freq', 0, 1024),
			'n_estimators'     : trial.suggest_int('n_estimators', 128, 128),
			
			'extra_trees'      : trial.suggest_categorical('extra_trees', [True, False]),
			'boosting_type'    : trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'rf']),
			'objective'        : trial.suggest_categorical('objective', objective),
			'metric'           : trial.suggest_categorical('metric', metric),
			'weight_adjustment': trial.suggest_categorical('weight_adjustment', [True, False]),
			'learning_rate'    : trial.suggest_categorical('learning_rate', [0.1]),
			'n_jobs'           : trial.suggest_categorical('n_jobs', [n_jobs]),
			'verbosity'        : trial.suggest_categorical('verbosity', [-1]),
			
			'reg_alpha'        : trial.suggest_float('reg_alpha', 1e-6, 128.0, log=True),
			'reg_lambda'       : trial.suggest_float('reg_lambda', 1e-6, 128.0, log=True),
			'min_split_gain'   : trial.suggest_float('min_split_gain', 1e-6, 128.0, log=True),
			'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.25, 1.0, step=0.05),
			'subsample'        : trial.suggest_float('subsample', 0.25, 0.95, step=0.05),
			'min_child_weight' : trial.suggest_float('min_child_weight', 1e-6, 1e6, log=True),
			'path_smooth'      : trial.suggest_float('path_smooth', 1e-6, 128.0, log=True),
			}
	
	if num_class > 1:
		parameters.update({'num_class': trial.suggest_categorical('num_class', [num_class])})
	
	return parameters


def lightgbm_short_parameters(trial):
	return {
			'n_estimators' : trial.suggest_int('n_estimators', 64, 1024, step=1),
			'learning_rate': trial.suggest_float('learning_rate', 0.001, 1.0, step=0.001),
			}


def sgdlinear_long_parameters(trial, loss, penalty, n_jobs, _type):
	parameters = {
			'loss'             : trial.suggest_categorical('loss', loss),
			'penalty'          : trial.suggest_categorical('penalty', penalty),
			'fit_intercept'    : trial.suggest_categorical('fit_intercept', [True, False]),
			'weight_adjustment': trial.suggest_categorical('weight_adjustment', [True, False]),
			'early_stopping'   : trial.suggest_categorical('early_stopping', [False]),
			'verbose'          : trial.suggest_categorical('verbose', [0]),
			
			'alpha'            : trial.suggest_float('alpha', 1e-6, 128.0, log=True),
			'l1_ratio'         : trial.suggest_float('l1_ratio', 0.0, 1.0, step=0.01),
			'epsilon'          : trial.suggest_float('epsilon', 1e-6, 128.0, log=True),
			}
	
	if _type.startswith('classification'):
		parameters.update(
				{
						'n_jobs'      : trial.suggest_categorical('n_jobs', [n_jobs]),
						'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
						}
				)
	
	return parameters


def sgdlinear_short_parameters(trial, _type):
	parameters = {
			'max_iter'      : trial.suggest_int('max_iter', 512, 2048, step=1),
			'early_stopping': trial.suggest_categorical('early_stopping', [False]),
			}
	
	if _type not in ['classification_multiclass']:
		parameters.update(
				{
						'n_iter_no_change'   : trial.suggest_int('n_iter_no_change', 2, 16, step=1),
						'validation_fraction': trial.suggest_float('validation_fraction', 0.05, 0.25, step=0.05),
						}
				)
	
	return parameters


def multinomialnb_parameters(trial):
	return {
			'force_alpha'      : trial.suggest_categorical('force_alpha', [True, False]),
			'fit_prior'        : trial.suggest_categorical('fit_prior', [True, False]),
			'weight_adjustment': trial.suggest_categorical('weight_adjustment', [True, False]),
			'alpha'            : trial.suggest_float('alpha', 1e-6, 128.0, step=0.1),
			}


def generate_hyperparameter_space(task_name, model_name, _type, trial, scoring, num_class, n_jobs, test):
	'''
	FIX
	structure
	trial warnings
	'''
	
	parameters = {}
	if (task_name, model_name, _type) == ('classification_binary', 'lightgbm', 'long'):
		objective = ['binary']
		metric = ['auc', 'average_precision', 'binary_logloss']
		parameters = lightgbm_long_parameters(trial, objective, metric, 1, n_jobs)
	
	elif (task_name, model_name, _type) == ('classification_binary', 'lightgbm', 'short'):
		parameters = lightgbm_short_parameters(trial)
	
	elif (task_name, model_name, _type) == ('classification_multiclass', 'lightgbm', 'long'):
		objective = ['multiclass'] if scoring in ['roc_auc_ovr', 'roc_auc_ova'] else ['multiclass', 'multiclassova']
		metric = ['auc_mu', 'multi_logloss', 'multi_error']
		parameters = lightgbm_long_parameters(trial, objective, metric, num_class, n_jobs)
	
	elif (task_name, model_name, _type) == ('classification_multiclass', 'lightgbm', 'short'):
		parameters = lightgbm_short_parameters(trial)
	
	elif (task_name, model_name, _type) == ('regression', 'lightgbm', 'long'):
		objective = ['regression', 'regression_l1', 'huber', 'fair', 'quantile', 'mape']
		metric = ['l1', 'l2', 'rmse', 'quantile', 'mape', 'huber', 'fair', 'poisson', 'tweedie']
		parameters = lightgbm_long_parameters(trial, objective, metric, 1, n_jobs)
	
	elif (task_name, model_name, _type) == ('regression', 'lightgbm', 'short'):
		parameters = lightgbm_short_parameters(trial)
	
	elif (task_name, model_name, _type) == ('classification_binary', 'sgdlinear', 'long'):
		loss = ['log_loss', 'modified_huber']
		penalty = ['l2', 'l1', 'elasticnet']
		parameters = sgdlinear_long_parameters(trial, loss, penalty, n_jobs, _type)
	
	elif (task_name, model_name, _type) == ('classification_binary', 'sgdlinear', 'short'):
		parameters = sgdlinear_short_parameters(trial, _type)
	
	elif (task_name, model_name, _type) == ('classification_multiclass', 'sgdlinear', 'long'):
		loss = ['modified_huber']
		penalty = ['l2', 'l1', 'elasticnet']
		parameters = sgdlinear_long_parameters(trial, loss, penalty, n_jobs, _type)
	
	elif (task_name, model_name, _type) == ('classification_multiclass', 'sgdlinear', 'short'):
		parameters = sgdlinear_short_parameters(trial, _type)
	
	elif (task_name, model_name, _type) == ('regression', 'sgdlinear', 'long'):
		loss = ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
		penalty = ['l2', 'l1', 'elasticnet']
		parameters = sgdlinear_long_parameters(trial, loss, penalty, n_jobs, _type)
	
	elif (task_name, model_name, _type) == ('regression', 'sgdlinear', 'short'):
		parameters = sgdlinear_short_parameters(trial, _type)
	
	elif (task_name, model_name, _type) == ('classification_multiclass', 'multinomialnb', 'long'):
		parameters = multinomialnb_parameters(trial)
	
	if not test and model_name == 'lightgbm' and _type == 'long':
		parameters = implement_lightgbm_production_mode(parameters, trial)
	
	return parameters
