import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold

from scripts.optimize import run_optimization


x_data, y_data = make_regression(n_samples=8096, n_features=128, n_informative=4)

x_data = pd.DataFrame(x_data)

cv = []
for train_index, test_index in KFold(n_splits=3, shuffle=True).split(x_data):
	cv.append((train_index, test_index))

best_hyperparameters_dictionary, weight_adjustment, important_features_list = run_optimization(
		task_name='regression', model_name='lightgbm', x_data=x_data, y_data=y_data, weight_data=[1] * len(y_data), cv=cv, n_trials_long=32, n_trials_short=16, patience=8, scoring=('neg_mean_squared_error', 'weighted'), calibration=None, n_jobs=16, drop_rate=0.50,
		min_columns_to_keep=8
		)

print(best_hyperparameters_dictionary, weight_adjustment, important_features_list)

# fix weighting
