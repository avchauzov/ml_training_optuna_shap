import optuna

from scripts.evaluate import optimize_hyperparameters
from scripts.feature_selection import get_select_features
from utils.optimization_utils import get_metric_dictionary


optuna.logging.set_verbosity(optuna.logging.INFO)


def run_optimization(task_name, model_name, x_data, y_data, cv, n_trials_long, n_trials_short, patience, scoring, calibration, n_jobs, drop_rate=0.50, min_columns_to_keep=8):
	"""
	Run a multi-step optimization process for hyperparameter tuning, feature selection, and more.

	Args:
		task_name (str): The machine learning task type (e.g., 'classification_binary', 'regression').
		model_name (str): The model type (e.g., 'lightgbm', 'sgdlinear').
		x_data (array-like): The input features for training.
		y_data (array-like): The target variable for training.
		cv (int): Number of cross-validation folds.
		n_trials_long (int): Number of trials for long optimization.
		n_trials_short (int): Number of trials for short optimization.
		patience (int): Patience parameter for optimization.
		n_jobs (int): Number of CPU cores to use for training.
		scoring (str): The scoring metric used for optimization.
		calibration (tuple): Tuple containing a flag and parameters for calibration.
		drop_rate (float): The drop rate for feature selection.
		min_columns_to_keep (int): Minimum number of columns to keep after feature selection.

	Returns:
		Various results and parameters based on the optimization steps.
	"""
	print('Step 1: Hyperparameters optimization')
	metric_dictionary = get_metric_dictionary(task_name)
	best_hyperparameters_dictionary, weight_adjustment = optimize_hyperparameters(x_data, y_data, cv, {}, n_jobs, n_trials_long, patience, scoring, 'long', task_name, model_name, metric_dictionary)
	
	if not best_hyperparameters_dictionary:
		raise Exception('Optimization failed: too few iterations')
	
	if task_name == 'classification_multiclass' and model_name == 'multinomialnb':
		return best_hyperparameters_dictionary, weight_adjustment
	
	print('Step 2: Feature selection')
	important_features_list = get_select_features(
			x_data, y_data, cv, best_hyperparameters_dictionary, weight_adjustment, drop_rate, min_columns_to_keep,
			scoring, task_name, model_name, metric_dictionary
			)
	
	print('Step 3: learning_rate and n_estimators optimization')
	best_hyperparameters_dictionary['weight_adjustment'] = weight_adjustment
	best_hyperparameters_dictionary, _ = optimize_hyperparameters(x_data[important_features_list], y_data, cv, best_hyperparameters_dictionary, n_jobs, n_trials_short, patience, scoring, 'short', task_name, model_name, metric_dictionary)
	
	return best_hyperparameters_dictionary, weight_adjustment, important_features_list
	
	'''if task in ['classification_binary', 'classification_multiclass'] and model == 'lightgbm':
		calibration_model_parameters = ('none', {})
		if calibration_parameters[0]:
			print('Step 4: Probability calibration')
			calibration_model_parameters = get_calibration_model(
					x_data[important_features_list], y_data, cv, best_hyperparameters_dictionary, weight_adjustment,
					n_jobs, calibration_parameters[1], scoring, metric_dictionary, task
					)

		return best_hyperparameters_dictionary, weight_adjustment, important_features_list, calibration_model_parameters
	else:
		return best_hyperparameters_dictionary, weight_adjustment, important_features_list'''
