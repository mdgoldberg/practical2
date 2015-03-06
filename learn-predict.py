import os
import sys
import json
from collections import Counter
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from sklearn import linear_model as lm
from sklearn import mixture as mix
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.kernel_approximation import RBFSampler
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.svm import SVC
from sklearn.tree import ExtraTreeClassifier as ETC
from sklearn.ensemble import RandomForestClassifier as RFC, GradientBoostingClassifier as GBC, BaggingClassifier as BC
from sklearn import cross_validation as CV

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

    print "reading test"
    with open(sys.argv[2], 'r') as tester:

        # Produce a CSV file.
        test_df = pd.read_csv(tester, delimiter=',', quotechar='"', index_col=0)
    print("done reading")

    X_test = test_df
    test_ids = test_df.index

    trainoutput = "trainpredictions.csv"
    testoutput = "testpredictions.csv"  # feel free to change this or take it as an argument


    # TODO train here, and learn your classification parameters
    print "learning..."
    # rbf_feature = RBFSampler(gamma=0.00075)#, random_state=1000)
    # X_features = rbf_feature.fit_transform(X_train)
    #X_features = X_train
    #model = lm.LogisticRegression(penalty='l1', dual=False, tol=1.0, C=100000.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None)
    #model = mix.GMM(n_components=1, covariance_type='diag', random_state=None, thresh=0.01, min_covar=0.001, n_iter=100, n_init=1, params='wmc', init_params='wmc')
    #model = MultinomialNB()
    #model = BernoulliNB()
    #model = SVC(C=1000,kernel='rbf')
    #model = KNC(n_neighbors=5, weights='distance', algorithm='auto', p=1)
    #model = RFC(n_estimators=50, n_jobs=-1, oob_score=True, criterion='gini', min_samples_split=5, max_leaf_nodes=100000) # this one is best so far
    #model = GBC(n_estimators=100, learning_rate=0.001, max_depth=4) # consistently 0.87 in CV
    #model = ETC()
    #model = KNC(n_neighbors=5, weights='distance', algorithm='auto', p=1)
    model = RFC(n_estimators=500, n_jobs=-1, criterion='gini') # this one is best so far
    #model = GBC(n_estimators=15, learning_rate=0.001, max_depth=3) # consistently 0.87 in CV
    
    model.fit(X_train, t_train) # was X_features
    print "done learning"
    print
    
    # test how well we're doing on the training data
    #cv_model = RFC(n_estimators=50, n_jobs=-1, oob_score=True, criterion='gini', min_samples_split=5, max_leaf_nodes=100000)
    #cv_model = MultinomialNB()
    #cv_model = KNC(n_neighbors=5, weights='distance', algorithm='auto', p=1)

    cv_model = RFC(n_estimators=500, n_jobs=-1, criterion='gini') # this one is best so far
    cv_scores = CV.cross_val_score(cv_model, X_train, t_train, cv=5)
    print 'cross-validation scores:', cv_scores
    print 'mean:', np.mean(cv_scores)
    
    print "making predictions..."

    # X_testfeatures = rbf_feature.transform(X_test)
    testpreds = model.predict(X_test) # was X_testfeatures
    #preds = np.argmax(X_test.dot(learned_W),axis=1)
    print "done making predictions"
    print
    
    print "writing predictions..."
    util.write_predictions(testpreds, test_ids, testoutput)
    print "done!"

if __name__ == "__main__":
    main()
    
