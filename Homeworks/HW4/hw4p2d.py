import numpy as np
from scipy.io import loadmat

def objective_value(beta, x, y):
	objective = 0 
	#print ("Dimensions of beta.T are %s\n" % beta.T.shape)
	for i in range(0, len(x)):
		objective += ( np.log(1 + np.exp(np.dot(beta.T, x[i]))) - y[i] * np.dot(beta.T, x[i]) )
	objective /= len(x)
	#print("Returning objective type:%s" % type(objective))
	return objective

def gradient(beta, x, y):
	grad = np.zeros((4)) 
	for i in range(0, len(x)):
		grad += ( ( x[i] / ( 1 + np.exp(np.dot(-beta.T, x[i])))) - y[i] *  x[i] )
		#print ("The grad for %s'th iteration is %s\n", (i, grad))
	grad /= len(x) 
	return grad
	
def is_power_two(count):
    if ((count & (count -1)) == 0 ):
        return True
    else:
        return False

def gradient_descent(beta, value, x, y, test_data, test_y):
    count = 1
    obj_val = objective_value(beta, x, y)
    grad = np.zeros((len(x)))
    best_holdout_error_rate = 100.0
    #print("In gradient_descent, obj_val is %s and value is %s" % (obj_val, value))
    while (obj_val > value):
        grad = gradient(beta, x, y)
        eta = 1.0
        while (objective_value(beta - eta * grad, x , y) >  obj_val - ( eta * (np.linalg.norm(grad)**2 )  / 2 )):
            eta = eta / 2
        beta -= eta * grad
        obj_val = objective_value(beta, x, y)
        #print('Objective value is %s, Count is %s' % (obj_val, count))
        
        if (is_power_two(count)):
            new_error_rate = 0.0
            for i in range(0, len(test_data)):
                temp = np.dot(beta.T, test_data[i])
                if (temp > 0 and test_y[i]==1):
                    pass
                elif (temp <= 0 and test_y[i]==0):
                    pass
                else:
                    new_error_rate += 1
            new_error_rate /= len(test_data)
            print('New error rate is %s and bestholderror_rate is %s and count is %s'%(new_error_rate, best_holdout_error_rate, count))
            print ('New error rate is %s' % new_error_rate)
            if ( (new_error_rate >  0.99 * best_holdout_error_rate ) and (count >= 32)):
                print ('Objective value is %s, Count is %s' % (obj_val, count))
                print ('Leaving after stopping condition has been found...')
                return count
            elif (new_error_rate < best_holdout_error_rate):
                best_holdout_error_rate = new_error_rate
                print('Best holdout error being updated')

        count += 1

    return count


def main():
	hw4data = loadmat('hw4data.mat')
	print ("Data has been loaded...")
	d = hw4data['data'].shape[1] 
	x = hw4data['data']
	col = np.ones((len(x)))
	x = np.column_stack([col, x])
	training_data = x[:3276,:]
	test_data = x[3276:,:]
	training_y = hw4data['labels'][:3276]
	test_y = hw4data['labels'][3276:]
	beta = np.zeros((d+1)) 
	value = -1000
	gradient_descent(beta, value, training_data, training_y, test_data, test_y)

if __name__ == main():
	main()
