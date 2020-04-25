    def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):
    N = X.shape[0]
    probabilities = compute_probabilities(X, theta, temp_parameter)
    selected_probabilities = np.choose(Y, probabilities)
    non_regulizing_cost = np.sum(np.log(selected_probabilities))
    non_regulizing_cost *= -1 / N
    regulizing_cost = np.sum(np.square(theta))
    regulizing_cost *= lambda_factor / 2.0
    return non_regulizing_cost + regulizing_cost
