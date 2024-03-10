"""
This script provides functions for training machine learning models including LightGBM, Multinomial Naive Bayes,
and Stochastic Gradient Descent (SGD) Linear models.
"""

from lightgbm import early_stopping, LGBMClassifier, LGBMRegressor, log_evaluation
from sklearn.linear_model import ElasticNet, LogisticRegression, SGDClassifier, SGDRegressor
from sklearn.naive_bayes import MultinomialNB


def train_lightgbm_model(model, data):
	data_train, data_test = data[: 3], data[3:]
	x_train, y_train, sample_weight_train = data_train
	x_test, y_test, sample_weight_test = data_test
	
	if model.boosting_type in ['dart']:
		model.fit(
				x_train, y_train,
				sample_weight=sample_weight_train,
				eval_set=[(x_test, y_test), (x_train, y_train)],
				eval_sample_weight=[sample_weight_test, sample_weight_train],
				eval_metric=model.metric,
				callbacks=[log_evaluation(period=-1, show_stdv=False)]
				)
	else:
		model.fit(
				x_train, y_train,
				sample_weight=sample_weight_train,
				eval_set=[(x_test, y_test), (x_train, y_train)],
				eval_sample_weight=[sample_weight_test, sample_weight_train],
				eval_metric=model.metric,
				callbacks=[
						early_stopping(int(model.n_estimators * 0.10), verbose=False),
						log_evaluation(period=-1, show_stdv=False)
						]
				)
	
	return model


def train_sklearn_model(model, data):
	x_train, y_train, sample_weight_train = data
	
	try:
		model.fit(x_train, y_train, sample_weight=sample_weight_train)
	
	except Exception as error:
		model = None
	
	return model


def train_any_model(model_name, data, hyperparameters, task_name):
	if model_name in ['lightgbm']:
		model = LGBMRegressor(**hyperparameters) if task_name == 'regression' else LGBMClassifier(**hyperparameters)
		return train_lightgbm_model(model, data)
	
	data = data[: 3]
	if model_name in ['multinomialnb']:
		model = MultinomialNB(**hyperparameters)
	
	elif model_name in ['sgdlinear']:
		model = SGDRegressor(**hyperparameters) if task_name == 'regression' else SGDClassifier(**hyperparameters)
	
	elif model_name in ['elasticnet']:
		model = ElasticNet(**hyperparameters)
	
	elif model_name in ['logisticregression']:
		model = LogisticRegression(**hyperparameters)
	
	else:
		return None
	
	return train_sklearn_model(model, data)
