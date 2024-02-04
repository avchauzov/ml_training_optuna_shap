from sklearn.linear_model import SGDClassifier, SGDRegressor


def train_model(data, hyperparameters, task_name):
	x_train, y_train, sample_weight_train = data
	
	if task_name == 'regression':
		model = SGDRegressor(**hyperparameters)
	
	else:
		model = SGDClassifier(**hyperparameters)
	
	model.fit(x_train, y_train, sample_weight=sample_weight_train)
	return model
