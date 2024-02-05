def update_dict_with_new_keys(space, hyperparameters):
	for key, value in hyperparameters.items():
		if key not in space:
			space[key] = value
	
	return space
