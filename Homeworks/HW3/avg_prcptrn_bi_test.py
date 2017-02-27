import numpy as np
import pandas as pd
import sklearn
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold

def load_data(training_file, training_size, test_file, test_size):
    print "Loading data..."
    data = pd.read_csv(training_file, nrows = training_size)
    test_data = pd.read_csv(test_file, nrows = test_size)
    return data, test_data

def model_counts(data, test_data):
    print "Building bigram model..."
    bigram_vectorizer = CountVectorizer(ngram_range=(1,2))
    X_train_counts = bigram_vectorizer.fit_transform(data['text'])
    X_test_counts = bigram_vectorizer.transform(test_data['text'])
    return X_train_counts, X_test_counts

def averaged_perceptron_train(X_train_counts, y):
    w = np.zeros((1, X_train_counts.shape[1]))
    total_w = np.zeros((1, X_train_counts.shape[1]))
    epochs = 2
    t, c = 0, 1
    total_t, total_c = 0, 1
    for i in range(0, epochs):
        for j in range(0, X_train_counts.shape[0]):
            if(y[j]*(w*X_train_counts[j].T+t)<=0):
                w = w + y[j]*X_train_counts[j]
                t = t + y[j]
            if i == 1 or ( i == 0 and j == X_train_counts.shape[1]-1):
                total_w += w*c
                total_t += t*c
                total_c += c
                c = c + 1
    weight = total_w / total_c
    avg_t = total_t / total_c
    return weight, avg_t

def averaged_perceptron_test(X_test_counts, weight, avg_t, yp):
    y = np.zeros((len(yp)))
    for i in range(0, len(yp)):
        y[i] = X_test_counts[i]*weight.T + avg_t
        if(y[i] <= 0):
            y[i] = -1
        else:
            y[i] = 1
    accuracy = np.mean(y==yp) * 100
    print 'The accuracy is %s %%\n' % accuracy
    return accuracy
    
def main():
    
    start_time = time.time()
    training_file, test_file = 'reviews_tr.csv', 'reviews_te.csv'
    training_size, test_size = 2000, 2000
    data, test_data = load_data(training_file, training_size, test_file, test_size)
    
    y = data['label'].values
    y[y==0] = -1 #converting 0s to -1 so make perceptron work
    yp = test_data['label'].values
    yp[yp==0] = -1
    
    X_train, X_test = data['text'], test_data['text'] 
    X_train_counts, X_test_counts = model_counts(X_train, X_test)
    weight, avg_t = averaged_perceptron_train(X_train_counts, y)
    prediction = averaged_perceptron_test(X_test_counts, weight, avg_t, yp)

#    for train_index, test_index in kf.split(data):
#        X_train, X_test = data[train_index[0]:train_index[-1]], data[test_index[0]:test_index[-1]]
#        X_train_counts, X_test_counts = model_counts(X_train, X_test)
#        weight, avg_t = averaged_perceptron_train(X_train_counts, y[train_index[0]:train_index[-1]])
#        predictions[i] = averaged_perceptron_test(X_test_counts, weight, avg_t, y[test_index[0]:test_index[-1]])
#        i = i + 1
    
#    print 'The average error rate is %s %%' % np.mean(predictions)
    print 'The seconds taken are %s' % ( time.time() - start_time )
    
if __name__ == '__main__':
    main()
