import numpy as np
from sklearn.metrics import average_precision_score

target = np.array([0, 0, 1, 1])
preds = np.array([0.1, 0.4, 0.35, 0.8])


sort_preds = np.argsort(preds)[::-1]
target_list = [0, 0, 1, 1]
count = 0.
for i in range(sort_preds.shape[0]):
    if len(target_list) == 0:
        break
    t = sort_preds[i]
    if t in target_list:
        target_list.remove(t)
        point = float((target.shape[0] - len(target_list))) / float(i + 1)
        count += point
        
my_ap = (count / target.shape[0] * 100.)
sk_ap = average_precision_score(target, preds)

print (my_ap, sk_ap)