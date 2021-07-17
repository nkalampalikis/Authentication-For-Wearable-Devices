#from data_iface import collect_points, make_sequential
import scipy as sp
import scipy.stats
from sklearn.metrics import f1_score, accuracy_score,confusion_matrix
import numpy as np

from data_extraction import collect_segment_data, make_sequential

# Given a model, some params, and a pile of labled data, evaluate the
# performance of the model.
def test(params, model, data_set, threshold, sequence_length):
    print("---------------------------------")
    print("Test Parameter Summary:")
    print("---------------------------------")
    print("Threshold:\t%0.2f\tSeq. Len.:\t%d" %(threshold,sequence_length))
    print("Win. Size:\t%d\tData Freq.:\t%d" %(params.window_sz,params.hz))
    print("Positive Data Set:")
    print(data_set["positive"])
    print("Negative Data Set:")
    print(data_set["negative"])
    print("=================================")
    # Load data specified by data set
    X = []
    Y = []
    print("Loading testing data...", end='',flush=True)
    for e in data_set["negative"]:
        points = collect_segment_data(params, e[0], [e[1]])
        points = make_sequential(params,points, sequence_length)
        for p in points:
            X.append(p)
            Y.append(0)

    for e in data_set["positive"]:
        points = collect_segment_data(params, e[0], [e[1]])
        points = make_sequential(params,points, sequence_length)
        for p in points:
            X.append(p)
            Y.append(1)
    print("done")

    print("Testing...", end='', flush=True)
    results = []
    y_pred_all = []
    for seq in X:
        y_pred = model.predict(seq)
        y_pred_all.append(y_pred[0][0])
        y_pred = np.array(list(map(lambda x: 1 if x > threshold else 0, y_pred)))
        if 1 in y_pred:
            results.append(1)
        else:
            results.append(0)
    assert(len(results) == len(Y))


    tr, fa, fr, ta = confusion_matrix(Y, results,labels=[0,1]).ravel()
    tar = (ta / (ta+fr)) * 100.
    trr = (tr / (tr+fa)) * 100.
    print("done\n")
    return [tar,trr,y_pred_all,Y]