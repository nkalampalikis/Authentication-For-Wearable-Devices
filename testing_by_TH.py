
from params import Parameters
from data_extraction import *
from copy import deepcopy
from train import Trainer
import os
import pathlib2


targets = [2, 3, 4, 5, 7, 8, 13, 15, 16, 19, 26, 28]
param = Parameters(5)

def train(target):
    others = deepcopy(targets)
    others.remove(target)
    t = Trainer(param, target, others)
    history = t.train(param)
    return t.model, history


def main():
    if not os.path.isdir("./log_training"):
        pathlib2.Path('./log_training').mkdir(parents=True, exist_ok=True)
    for t in targets:
        trained_model = train(t)
        hist = trained_model[1]
        log_file = "./log_training/{}.log".format(t)
        with open(log_file, 'w')as f:
            f.write(str(t) + '='+ str(hist.history['acc']))
        print("DONE "+str(t))





if __name__ == "__main__":
    main()