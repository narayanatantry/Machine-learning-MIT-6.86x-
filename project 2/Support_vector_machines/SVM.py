def one_vs_rest_svm(train_x, train_y, test_x):
    clf = LinearSVC(C=0.1, random_state=0)
    clf.fit(train_x, train_y)
    pred_test_y = clf.predict(test_x)
    return pred_test_y
