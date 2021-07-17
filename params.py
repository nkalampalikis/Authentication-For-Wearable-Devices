from db import DB
import os
class Parameters:
    def __init__(self, window_size=5):
        # Data options
        self.window_sz = window_size
        self.hz = 50
        #BCG MODEL
        #self.model = {'conv3_enabled': False, 'conv2_enabled': True, 'dense2_neurons': 128, 'conv3_activation': 'tanh', 'dense1_activation': 'tanh', 'dense1_dropout': 0.4, 'dense2_enabled': True, 'dense3_neurons': 512, 'conv1_neurons': 16, 'conv2_activation': 'selu', 'convdense_dropout': 0.2, 'conv4_enabled': True, 'dense3_dropout': 0.2, 'conv4_neurons': 64, 'conv5_activation': 'sigmoid', 'dense1_neurons': 128, 'dense3_enabled': True, 'loss': 'mae', 'dense2_activation': 'relu', 'conv3_neurons': 256, 'conv1_activation': 'relu', 'final_activation': 'tanh', 'conv4_activation': 'relu', 'conv2_neurons': 256, 'dense3_activation': 'tanh', 'optimizer': 'adagrad', 'dense1_enabled': True, 'conv5_neurons': 64, 'dense2_dropout': 0, 'conv5_enabled': True}

        # BCG MODEL
        self.model ={'conv5_neurons': 128, 'dense3_enabled': False, 'dense2_dropout': 0.2, 'dense1_dropout': 0.2,
         'conv5_enabled': False, 'dense2_activation': 'sigmoid', 'conv2_neurons': 256, 'loss': 'mse',
         'convdense_dropout': 0.2, 'conv3_enabled': True, 'conv4_activation': 'selu', 'optimizer': 'sgd',
         'final_activation': 'tanh', 'conv3_neurons': 256, 'conv2_activation': 'tanh', 'dense1_enabled': True,
         'dense1_activation': 'tanh', 'dense1_neurons': 512, 'conv4_enabled': True, 'conv5_activation': 'sigmoid',
         'conv1_neurons': 8, 'conv4_neurons': 256, 'dense2_enabled': False, 'conv1_activation': 'tanh',
         'dense3_activation': 'tanh', 'conv3_activation': 'selu', 'dense3_neurons': 128, 'conv2_enabled': True,
         'dense3_dropout': 0.5, 'dense2_neurons': 128}

        # t=7
        # log_file = "{}.log".format(t)
        # #with open(log_file, 'r') as f:
        #     #file = f.readlines()
        #     #print(file)
        #     #dict = file[0].split("{},':")
        #     #dict = str( dict.split(","))
        #     #dict= dict.split("'")
        #    # for val in dict:
        #     #    print(val)



        self.epochs = 100
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.db = DB(dir_path + "/databases_BVP/db_" + str(window_size) + ".sqlite", init=False)

