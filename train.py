from model import Model
import time
import numpy as np
from data_extraction import collect_segment_data
from params import *


class Trainer:
    def __init__(self, params, owner, others):
        # Use first four segments of the first session (~8 minutes of data)
        self.train_data_format = [
            # (session, segment)
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
        ]

        # Acquire model (either create or load)
        print("==> Acquiring model...", end='', flush=True)
        self.weights_path = "./models_cache/%d/%d/weights.h5" % (params.window_sz, owner)
        self.topo_path = "./models_cache/%d/%d/topo.json" % (params.window_sz, owner)
        self.model = Model.get( params, self.topo_path, self.weights_path)
        print("done")

        # Assemble training data
        print("==> Acquiring raw data...", end='', flush=True)

        pos = collect_segment_data(params, owner, self.train_data_format)
        neg = []
        print(type(others))
        for o in others:
            neg += collect_segment_data(params, o, self.train_data_format)
        print("done")

        # Format data into correctly shaped numpy tensors
        self.train_tensor = np.ndarray((
            len(pos)+len(neg),
            2,
            3,
            params.window_sz*params.hz
        ))
        self.train_labels = []

        print("==> Formatting tensor...", end='', flush=True)
        for r in range(0, len(neg)):
            self.train_tensor[r] = neg[r]
            self.train_labels.append(0)
        for r in range(0, len(pos)):
            self.train_tensor[r+len(neg)] = pos[r]
            self.train_labels.append(1)
        self.train_labels = np.array(self.train_labels)
        print("done")

        # Generate data weights
        print("==> Generating weights...", end='', flush=True)
        self.weights = {
            0 : 1.0,
            1 : float(len(neg)/len(pos))
        }
        print("done")

    def train(self, param):
        print("Training...", end='', flush=True)
        start_time = time.time()
        history = self.model.fit(
                self.train_tensor, self.train_labels,
                batch_size=32,
                epochs=param.epochs,
                verbose=0,
                class_weight=self.weights,
                shuffle=True
            )
        print("done")
        end_time = time.time()
        Model.save_model(self.model, self.topo_path, self.weights_path)
        print("Model trained in %0.2fs" % (end_time - start_time))
        return history