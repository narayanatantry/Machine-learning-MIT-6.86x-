def update_y(train_y, test_y):
    return np.remainder(train_y, 3), np.remainder(test_y, 3)

def compute_test_error_mod3(X, Y, theta, temp_parameter):
    assigned_labels = get_classification(X, theta, temp_parameter)
    return 1 - np.mean(np.remainder(assigned_labels,3) == Y)
