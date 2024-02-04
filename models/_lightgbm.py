import lightgbm


def train_model(data_train, data_test, hyperparameters, task_name):
	x_train, y_train, sample_weight_train = data_train
	x_test, y_test, sample_weight_test = data_test
	
	if task_name == 'regression':
		model = lightgbm.LGBMRegressor(**hyperparameters)
	
	else:
		model = lightgbm.LGBMClassifier(**hyperparameters)
	
	if hyperparameters['boosting_type'] in ['dart']:
		model.fit(
				x_train, y_train,
				sample_weight=sample_weight_train,
				eval_set=[(x_test, y_test), (x_train, y_train)],
				eval_sample_weight=[sample_weight_test, sample_weight_train],
				eval_metric=hyperparameters['metric'],
				callbacks=[lightgbm.log_evaluation(period=-1, show_stdv=False)]
				)
	
	else:
		model.fit(
				x_train, y_train,
				sample_weight=sample_weight_train,
				eval_set=[(x_test, y_test), (x_train, y_train)],
				eval_sample_weight=[sample_weight_test, sample_weight_train],
				eval_metric=hyperparameters['metric'],
				callbacks=[
						lightgbm.early_stopping(int(hyperparameters['n_estimators'] * 0.10), verbose=False),
						lightgbm.log_evaluation(period=-1, show_stdv=False)
						]
				)
	
	return model
