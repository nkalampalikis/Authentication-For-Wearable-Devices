from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve , auc
from model import Model
from test_data_sets.first_sessions import prepareFirst
from test_data_sets.second_sessions import prepareSecond
from test_data_sets.third_sessions import prepareThird
from test import test
from params import Parameters
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d



p= Parameters(5)
sequence_length = 1
targets = [2, 3, 4, 5, 7, 8, 15, 16, 19, 26, 28]
lineStyleList = ["-", "--", "-.", ":"]


def testing(model, target, threshold, test_set):

    if test_set == 1:
        data_set = prepareFirst(target)
    elif test_set == 2:
        data_set = prepareSecond(target)
    elif test_set == 3:
        data_set = prepareThird(target)

    sequence_length = 1

    results = test(
        p,
        model,
        data_set,
        threshold,
        sequence_length
    )
    return results

def main():


        #i=2
    for i in range(1,4):
        plt.figure()
        plt.xlim([0, 100])
        plt.ylim([0, 100])
        plt.xlabel('False Accept Rate (%)')
        plt.ylabel('True Accept Rate (%)')

        tprs = []
        fprs = []
        mean_fpr = np.linspace(0, 1, 100)
        rocValues = {}
        line = 0
        legends =[]
        for t in targets:
            print("TARGET", t)
            print("==> Acquiring model...", end='', flush=True)
            topo_path = "./models_cache_bvp/5/{}/topo.json".format(t)
            weights_path = "./models_cache_bvp/5/{}/weights.h5".format(t)
            model = Model.get(p, topo_path, weights_path)
            results = testing(model,t, 1000,i)

            fpr, tpr, thresholds = roc_curve(results[3], results[2])
            rocValues[t] = [fpr, tpr, thresholds]
            tprs.append(np.interp(mean_fpr, fpr, tpr))

            print("Target", t)

            for item in results[3]:
                if item > 1 or item < 0 or np.isnan(item):
                    print("Invalid item", item)

            for item in results[2]:
                if np.isnan(item):
                    print("Invalid item", item)


            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            # for rocValue in range(0, len(fpr)):
            #     fpr[rocValue] = fpr[rocValue] * 100
            #     tpr[rocValue] = tpr[rocValue] * 100
            fprs.append(fpr)
            test_auc = auc(fpr, tpr)
            tuple = []
            tuple.append(test_auc)
            tuple.append(eer * 100)
            plt.plot(100*fpr, 100*tpr, lineStyle = lineStyleList[line%len(lineStyleList)],label='%s : AUC %0.8f; EER %0.8f%s' % ( t,test_auc, eer * 100, '%'))
            legends.append(tuple)
            line = line +1

        tprs[-1][0] = 0.0
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1
        xs =mean_fpr
        ys =np.mean(tprs, axis=0)


        eer = brentq(lambda x: 1. - x - interp1d(xs, ys)(x), 0., 1.)
        test_auc = auc(xs, ys)

        plt.plot(xs* 100,ys * 100, lw=4, color='black', ls="dotted",label='AVG : AUC %0.8f; EER %0.8f%s' % ( test_auc, eer * 100, '%'))
        avg_legend =[]
        avg_legend.append(test_auc)
        avg_legend.append(eer * 100)
        legends.append(avg_legend)

        targetlabels = ["Participant 2" , "Participant 3", "Participant 4", "Participant 5","Participant 7", "Participant 8",  "Participant 15", "Participant 16", "Participant 19", "Participant 26","Participant 28", "Average"]
        targetlabels_new = []
        legend = 0
        for targ in targetlabels:
            targetlabels_new.append(targ +', AVG AUC %0.8f; EER %0.8f%s' % ( legends[legend][0], legends[legend][1], '%'))
            legend = legend +1


        plt.legend(targetlabels_new)
        plt.title("BVP External Set, Session "+str(i))
        plt.show()




if __name__ == "__main__":
     main()