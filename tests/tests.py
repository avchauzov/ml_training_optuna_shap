import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold

from scripts.optimize import run_optimization


def test_lightgbm_classification_binary():
	x_data, y_data = make_classification(n_samples=8096, n_features=128, n_informative=4)
	
	x_data = pd.DataFrame(x_data)
	
	cv = []
	for train_index, test_index in StratifiedKFold(n_splits=3, shuffle=True).split(x_data, y_data):
		cv.append((train_index, test_index))
	
	best_hyperparameters_dictionary, weight_adjustment, important_features_list = run_optimization(
			task_name='classification_binary', model_name='lightgbm', x_data=x_data, y_data=y_data, weight_data=[1] * len(y_data), cv=cv, n_trials_long=32, n_trials_short=16, patience=8, scoring=('roc_auc', 'weighted'), calibration=None, n_jobs=16, drop_rate=0.50,
			min_columns_to_keep=8
			)
	
	print(best_hyperparameters_dictionary, weight_adjustment, important_features_list)
	
	assert all(
			key in ['n_estimators', 'learning_rate', 'num_leaves', 'max_depth', 'min_child_samples', 'max_bin', 'min_data_in_bin', 'max_cat_threshold', 'subsample_freq', 'extra_trees', 'boosting_type', 'objective', 'metric', 'reg_alpha', 'reg_lambda', 'min_split_gain', 'colsample_bytree', 'subsample', 'min_child_weight',
			        'path_smooth', 'n_jobs', 'verbosity', 'device', 'gpu_use_dp'] for key in list(best_hyperparameters_dictionary.keys())
			)
	
	assert weight_adjustment in [True, False]
	
	assert len(important_features_list) <= 128
	assert len(important_features_list) >= 1
