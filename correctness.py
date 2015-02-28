import pandas as pd
import sys
import json

if len(sys.argv) >= 2:
    csv_fn = sys.argv[1]
else:
    print 'usage: python correctness.py train_predictions.csv'
    sys.exit(0)

pred_df = pd.read_csv(csv_fn, index_col=0, engine='c')

with open('train_classes.json', 'r') as f:
    actual_dict = json.load(f)

assert actual_dict.keys() == list(pred_df.index)

total = 0
correct = 0.
for key, actual in actual_dict.iteritems():
    total += 1
    if actual == pred_df.loc[key, 'Prediction']:
        correct += 1

print correct / total
