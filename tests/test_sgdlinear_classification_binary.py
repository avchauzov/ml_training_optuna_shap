import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold

from scripts.optimize import run_optimization


x_data, y_data = make_classification(n_samples=8096, n_features=128, n_informative=4)

x_data = pd.DataFrame(x_data)

cv = []
for train_index, test_index in StratifiedKFold(n_splits=3, shuffle=True).split(x_data, y_data):
	cv.append((train_index, test_index))

best_hyperparameters_dictionary, weight_adjustment, important_features_list = run_optimization(
		task_name='classification_binary', model_name='sgdlinear', x_data=x_data, y_data=y_data, cv=cv, n_trials_long=32, n_trials_short=16, patience=8, scoring=('roc_auc', 'weighted'), calibration=None, n_jobs=16, drop_rate=0.25,
		min_columns_to_keep=8
		)

print(best_hyperparameters_dictionary, weight_adjustment, important_features_list)
