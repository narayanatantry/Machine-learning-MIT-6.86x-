import numpy
def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    ys = feature_matrix @ theta + theta_0
    loss = np.maximum(1 - ys * labels, np.zeros(len(labels)))
    return np.mean(loss)
