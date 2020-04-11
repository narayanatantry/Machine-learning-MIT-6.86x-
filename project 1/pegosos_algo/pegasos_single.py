def pegasos_single_step_update(feature_vector,label,L,eta,current_theta,0current_theta_0):
    mult = 1 - (eta * L)
    if label * (np.dot(feature_vector, current_theta) + current_theta_0) <= 1:
        return ((mult * current_theta) + (eta * label * feature_vector),
                (current_theta_0) + (eta * label))
    return (mult * current_theta, current_theta_0)
