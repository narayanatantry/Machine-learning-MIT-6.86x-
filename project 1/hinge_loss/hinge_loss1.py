import numpy
def hinge_loss_single(feature_vector , label ,theta, theta_0):
    y = theta @ feature_vector + theta_0
    return max(0,1-y*label)



#or
import numpy
def hinge_loss_single(feature_vector , label ,theta, theta_0):
    y = np.dot(theta, feature_vector)+theta_0
    loss = max(0.0,1-y*label)
    return(loss)
