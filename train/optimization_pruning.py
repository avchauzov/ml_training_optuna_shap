import optuna

from train.cross_validation import cross_validate


def apply_pruning(data, cv, space, best_score, patience, task_name, model_name, metric_name, trial):
	x_data, y_data, weight_data = data
	
	try:
		best_value = trial.study.best_value
		best_score.append(best_value)
	
	except ValueError as _:
		pass
	
	if len(best_score) >= patience:
		if best_score[-1] == best_score[-patience]:
			raise optuna.TrialPruned()
	
	return cross_validate([x_data, y_data, weight_data], cv, space, task_name, model_name, metric_name)


def is_model_pruned(hyperparameters):
	if hyperparameters.get('num_leaves', 0) > (2 ** hyperparameters.get('max_depth', 0) * 0.75):
		return True
	
	return False
