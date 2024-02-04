from settings.settings import METRICS, TASKS


def get_metric_dictionary_from_settings(task_name):
	metric_dictionary = {}
	for metric_name in TASKS[task_name]:
		metric_dictionary[metric_name] = METRICS[metric_name]
	
	return metric_dictionary
