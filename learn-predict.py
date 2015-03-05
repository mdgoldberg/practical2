import os
from collections import Counter
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
#from scipy import sparse
from sklearn import linear_model as lm
import pandas as pd
import sys
import json
from sklearn import mixture as mix
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.kernel_approximation import RBFSampler
from sklearn.neighbors import KNeighborsClassifier


import util

## The following function does the feature extraction, learning, and prediction
def main():
    print "reading train"
    with open(sys.argv[1], 'r') as trainer:

        # Produce a CSV file.
        train_df = pd.read_csv(trainer, delimiter=',', quotechar='"', index_col=0)
    print("done reading")

    X_train = train_df
    train_ids = train_df.index

    with open('train_classes.json', 'r') as t_classes:
        classes = json.load(t_classes)
    t_train = [classes[ID] for ID in train_ids]
    #print train_ids

    print "reading test"
    with open(sys.argv[2], 'r') as tester:

        # Produce a CSV file.
        test_df = pd.read_csv(tester, delimiter=',', quotechar='"', index_col=0)
    print("done reading")

    X_test = test_df
    test_ids = test_df.index

    #train_dir = "train"
    #test_dir = "test"
    trainoutput = "trainpredictions.csv"
    testoutput = "testpredictions.csv"  # feel free to change this or take it as an argument

    #X_train,global_feat_dict,t_train,train_ids = extract_feats(ffs, train_dir)

    
    # TODO train here, and learn your classification parameters
    print "learning..."
    #rbf_feature = RBFSampler(gamma=0.00075, random_state=1000)
    #X_features = rbf_feature.fit_transform(X_train)
    X_features = X_train
    #model = lm.LogisticRegression(penalty='l1', dual=False, tol=1.0, C=100000.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None)
    #model = lm.BernoulliRBM(n_components=256, learning_rate=0.1, batch_size=10, n_iter=10, verbose=0, random_state=None)
    #model = mix.GMM(n_components=1, covariance_type='diag', random_state=None, thresh=0.01, min_covar=0.001, n_iter=100, n_init=1, params='wmc', init_params='wmc')
    #model = MultinomialNB(alpha=0.0000000001, fit_prior=True, class_prior=None)
    #model = BernoulliNB(alpha=0.0000000001,binarize=0.1, fit_prior=True, class_prior=None)
    model = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None)
    model.fit(X_features, t_train)
    #learned_W = np.random.random((len(global_feat_dict),len(util.malware_classes)))
    print "done learning"
    print

    trainpreds = model.predict(X_features)

    util.write_predictions(trainpreds, train_ids, trainoutput)
    
    # get rid of training data and load test data
    #del X_train
    #del t_train
    #del train_ids
    print "extracting test features..."
    #X_test,_,t_ignore,test_ids = extract_feats(ffs, test_dir, global_feat_dict=global_feat_dict)
    print "done extracting test features"
    print
    
    # TODO make predictions on text data and write them out
    print "making predictions..."
    #X_testfeatures = rbf_feature.transform(X_test)
    X_testfeatures = X_test
    testpreds = model.predict(X_testfeatures)
    #preds = np.argmax(X_test.dot(learned_W),axis=1)
    print "done making predictions"
    print
    
    print "writing predictions..."
    util.write_predictions(testpreds, test_ids, testoutput)
    print "done!"

if __name__ == "__main__":
    main()
    