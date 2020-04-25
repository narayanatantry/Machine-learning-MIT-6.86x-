
def project_onto_PC(X, pcs, n_components, feature_means):

    centered_data = X - feature_means
    return np.dot(centered_data, pcs[:, range(n_components)])
