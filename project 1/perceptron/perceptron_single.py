import numpy
def perceptron_single_step_update(feature_vector, label, current_theta, current_theta_0):
    if label * (np.dot(current_theta, feature_vector) + current_theta_0) <= 1e-7:
        return (current_theta + label * feature_vector, current_theta_0 + label)
    return (current_theta, current_theta_0)
