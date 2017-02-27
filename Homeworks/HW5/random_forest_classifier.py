import numpy as np
import pandas as pd
import sklearn
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import KFold

def load_data(training_file, training_size, test_file, test_size):
    print "Loading data..."
    data = pd.read_csv(training_file, nrows = training_size)
    test_data = pd.read_csv(test_file, nrows = test_size)
    return data, test_data

def model_counts(data, test_data):
    print "Building unigram model..."
    count_vect = CountVectorizer(ngram_range=(1,2), min_df=0.005, max_df=0.99)
    X_train_counts = count_vect.fit_transform(data['text'])
    X_test_counts = count_vect.transform(test_data['text'])
    return X_train_counts, X_test_counts

def classifier(X_train_counts, data):
    print "Buliding Naive Bayes Classifier..."
    text_clf = RandomForestClassifier(n_jobs=2, n_estimators=500, max_features="auto", oob_score=True).fit(X_train_counts, data['label'])
    return text_clf

def prediction(X_test_counts, text_clf, test_data, start_time):
    print "Predicting on the test set..."
    predicted = text_clf.predict(X_test_counts)
    accuracy = ( np.mean(predicted == test_data['label'])*100)
    print "Accuracy is %s %%" % accuracy
    print ("The code took %s seconds...\n" % (time.time()-start_time))
    return accuracy
    

def main():

    training_file, test_file = 'reviews_tr.csv', 'reviews_te.csv'
    training_size, test_size = 20000, 20000
    data, test_data = load_data(training_file, training_size, test_file, test_size)
    
    folds = 5
    kf = KFold(n_splits=folds)
    predictions = np.zeros((folds))
    i = 0
    """
    for train_index, test_index in kf.split(data):
        start_time = time.time()
        #print train_index, test_index
        X_train, X_test = data[train_index[0]:train_index[-1]], data[test_index[0]:test_index[-1]]
        X_train_counts, X_test_counts = model_counts(X_train, X_test)
        text_clf = classifier(X_train_counts, X_train)
        predictions[i] = prediction(X_test_counts, text_clf, X_test, start_time)
        i = i + 1
    """
    start_time = time.time()
    X_train, X_test = data, test_data
    X_train_counts, X_test_counts = model_counts(X_train, X_test)
    text_clf = classifier(X_train_counts, X_train)
    predictions[i] = prediction(X_test_counts, text_clf, X_test, start_time)
    print np.mean(predictions)
    
    print('Printing train error now')
    predictions[i] = prediction(X_train_counts, text_clf, X_train, start_time)
    print np.mean(predictions)


if __name__ == '__main__':
    main()
