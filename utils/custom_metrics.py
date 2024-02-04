import numpy as np


def symmetrical_mape(y_true, y_pred, sample_weight):
	numerator = np.nansum(np.multiply(sample_weight, np.abs(y_true - y_pred)))
	denominator = np.nansum(np.multiply(sample_weight, np.abs(y_true) + np.abs(y_pred))) + 1e-7
	
	return 1 / len(y_true) * numerator / denominator
