import lightgbm
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.naive_bayes import MultinomialNB


def train_lightgbm_model(data_train, data_test, hyperparameters, task_name):
	x_train, y_train, sample_weight_train = data_train
	x_test, y_test, sample_weight_test = data_test
	
	if task_name in ['regression']:
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


def train_multinomialnb_model(data, hyperparameters):
	x_train, y_train, sample_weight_train = data
	
	model = MultinomialNB(**hyperparameters)
	model.fit(x_train, y_train, sample_weight=sample_weight_train)
	return model


def train_sgdlinear_model(data, hyperparameters, task_name):
	x_train, y_train, sample_weight_train = data
	
	if task_name in ['regression']:
		model = SGDRegressor(**hyperparameters)
	
	else:
		model = SGDClassifier(**hyperparameters)
	
	model.fit(x_train, y_train, sample_weight=sample_weight_train)
	return model
