import sys
import os
import json
import util
import features
import numpy as np
import pandas as pd
from scipy import sparse
from collections import Counter
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


def extract_feats(ffs, direc="train", global_feat_dict=None):
    """
    arguments:
      ffs are a list of feature-functions.
      direc is a directory containing xml files (expected to be train or test).
      global_feat_dict is a dictionary mapping feature_names to column-numbers; it
      should only be provided when extracting features from test data, so that 
      the columns of the test matrix align correctly.

    returns: 
      a sparse design matrix, a dict mapping features to column-numbers,
      a vector of target classes, and a list of system-call-history ids in order 
      of their rows in the design matrix.
      
      Note: the vector of target classes returned will contain the true indices of the
      target classes on the training data, but will contain only -1's on the test
      data
    """
    fds = [] # list of feature dicts
    ids = [] 
    c_none = Counter()
    a_none = Counter()
    c_mal = Counter()
    a_mal = Counter()
    i = 1
    for datafile in os.listdir(direc):
        if i % 50 == 0:
            print 'processing file ', i
        i += 1
        # extract id and true class (if available) from filename
        id_str,clazz = datafile.split('.')[:2]
        ids.append(id_str)
        rowfd = {}
        # parse file as an xml document
        tree = ET.parse(os.path.join(direc,datafile))
        # for all_sec in tree.iter('all_section'):
        #     for syscall in all_sec:
        #         if syscall.tag == 'vm_protect':
        #             print clazz, syscall.attrib
        # accumulate features
        for ff in ffs:
            rowfd.update(ff(tree, direc + '/' + datafile))
        fds.append(rowfd)

    # print 'safe syscalls:'
    # print c_none.most_common()
    # print 'malicious syscalls:'
    # print c_mal.most_common()
    # print 'safe syscall-attributes:'
    # print a_none.most_common()
    # print 'malicious syscall-attributes:'
    # print a_mal.most_common()

    X,feat_dict = make_design_mat(fds,global_feat_dict)
    return X, feat_dict, ids


def make_design_mat(fds, global_feat_dict=None):
    """
    arguments:
      fds is a list of feature dicts (one for each row).
      global_feat_dict is a dictionary mapping feature_names to column-numbers; it
      should only be provided when extracting features from test data, so that 
      the columns of the test matrix align correctly.
       
    returns: 
        a sparse NxD design matrix, where N == len(fds) and D is the number of
        the union of features defined in any of the fds 
    """
    if global_feat_dict is None:
        all_feats = set()
        [all_feats.update(fd.keys()) for fd in fds]
        feat_dict = dict([(feat, i) for i, feat in enumerate(sorted(all_feats))])
    else:
        feat_dict = global_feat_dict
        
    cols = []
    rows = []
    data = []        
    for i in xrange(len(fds)):
        temp_cols = []
        temp_data = []
        for feat,val in fds[i].iteritems():
            try:
                # update temp_cols iff update temp_data
                temp_cols.append(feat_dict[feat])
                temp_data.append(val)
            except KeyError as ex:
                if global_feat_dict is not None:
                    pass  # new feature in test data; nbd
                else:
                    raise ex

        # all fd's features in the same row
        k = len(temp_cols)
        cols.extend(temp_cols)
        data.extend(temp_data)
        rows.extend([i]*k)

    assert len(cols) == len(rows) and len(rows) == len(data)
   

    X = sparse.csr_matrix((np.array(data),
                   (np.array(rows), np.array(cols))),
                   shape=(len(fds), len(feat_dict)))
    return X, feat_dict


if len(sys.argv) >= 3:
    out_csv = sys.argv[1]
    out_json = sys.argv[2]
else:
    print 'usage: python extract_test.py outfile.csv outfile.json'
    sys.exit(0)

print 'extracting features'
X_train, global_feat_dict, ids = extract_feats(features.ffs, 'train')
print 'converting to dense matrix'
X_train = X_train.todense()
inverted_feat_dict = {v: k for k, v in global_feat_dict.items()}
print 'converting to dataframe'
phi = pd.DataFrame(X_train, index=ids, columns=[inverted_feat_dict[i] for i in
    xrange(len(inverted_feat_dict))])

print 'writing to csv'
phi.to_csv(out_csv, encoding='utf-8')
print 'writing to json'
with open(out_json, 'w') as f:
    json.dump(global_feat_dict, f)
