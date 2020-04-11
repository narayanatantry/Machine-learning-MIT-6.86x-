import numpy
def average_perceptron(feature_matrix, labels, T):
     (nsamples, nfeatures) = feature_matrix.shape
    theta = np.zeros(nfeatures)
    theta_sum = np.zeros(nfeatures)
    theta_0 = 0.0
    theta_0_sum = 0.0
    for t in range(T):
        for i in get_order(nsamples):
            theta, theta_0 = perceptron_single_step_update(
                feature_matrix[i], labels[i], theta, theta_0)
            theta_sum += theta
            theta_0_sum += theta_0
    return (theta_sum / (nsamples * T), theta_0_sum / (nsamples * T))
