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
    model = lm.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None)
    model.fit(X_train, t_train)
    #learned_W = np.random.random((len(global_feat_dict),len(util.malware_classes)))
    print "done learning"
    print

    #trainpreds = model.predict(X_train)

    #util.write_predictions(trainpreds, train_ids, trainoutput)
    
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
    testpreds = model.predict(X_test)
    #preds = np.argmax(X_test.dot(learned_W),axis=1)
    print "done making predictions"
    print
    
    print "writing predictions..."
    util.write_predictions(testpreds, test_ids, testoutput)
    print "done!"

if __name__ == "__main__":
    main()
    