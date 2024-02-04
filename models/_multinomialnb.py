from sklearn.naive_bayes import MultinomialNB


def train_model(data, hyperparameters):
	x_train, y_train, sample_weight_train = data
	
	model = MultinomialNB(**hyperparameters)
	model.fit(x_train, y_train, sample_weight=sample_weight_train)
	return model
