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
	

def gradient_descent(beta, x, y, value):
	count = 0
	obj_val = objective_value(beta, x, y)
	grad = np.zeros((len(beta)))
	#print("In gradient_descent, obj_val is %s and value is %s" % (obj_val, value))
	while (obj_val > value):
		grad = gradient(beta, x, y)
		eta = 1.0
		while (objective_value(beta - eta * grad, x , y) >  obj_val  - ((eta * (np.linalg.norm(grad)**2 )/2) )):
			eta = eta / 2

                beta -= eta * grad
		count = count + 1
		obj_val = objective_value(beta, x, y)
		print('Objective value is %s, Count is %s' % (obj_val, count))
	return count


def main():
	hw4data = loadmat('hw4data.mat')
	print ("Data has been loaded...")
	d = hw4data['data'].shape[1] 
	x = hw4data['data']
	#x = x - x.mean(axis=0)
	#x = x / np.std(x, axis=0)
	#x = x / np.max(x, axis = 0)
	col = np.ones((len(x)))
	x = np.column_stack([col, x])
	y = hw4data['labels']
	beta = np.zeros((d+1)) 
	value = .65064
	gradient_descent(beta, x, y, value)

if __name__ == main():
	main()
