from random import randint
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import BatchNormalization

traits_dict = {
    # Convolution layers
    # We will always have at least one conv layer
    "conv1_neurons"     : [8,16,32,64,128,256],
    "conv1_activation"  : ["relu", "selu", "sigmoid", "tanh"],

    "conv2_enabled"     : [True,False],
    "conv2_neurons"     : [8,16,32,64,128,256],
    "conv2_activation"  : ["relu", "selu", "sigmoid", "tanh"],

    "conv3_enabled"     : [True,False],
    "conv3_neurons"     : [8,16,32,64,128,256],
    "conv3_activation"  : ["relu", "selu", "sigmoid", "tanh"],

    "conv4_enabled"     : [True,False],
    "conv4_neurons"     : [8,16,32,64,128,256],
    "conv4_activation"  : ["relu", "selu", "sigmoid", "tanh"],

    "conv5_enabled"     : [True,False],
    "conv5_neurons"     : [8,16,32,64,128,256],
    "conv5_activation"  : ["relu", "selu", "sigmoid", "tanh"],

    "convdense_dropout" : [0, 0.2, 0.4, 0.5],

    # Dense layers
    "dense1_enabled"    : [True,False],
    "dense1_neurons"    : [16,32,64,128,256,512],
    "dense1_activation" : ["relu", "selu", "sigmoid", "tanh"],
    "dense1_dropout"    : [0, 0.2, 0.4, 0.5],

    "dense2_enabled"    : [True,False],
    "dense2_neurons"    : [16,32,64,128,256,512],
    "dense2_activation" : ["relu", "selu", "sigmoid", "tanh"],
    "dense2_dropout"    : [0, 0.2, 0.4, 0.5],

    "dense3_enabled"    : [True,False],
    "dense3_neurons"    : [16,32,64,128,256,512],
    "dense3_activation" : ["relu", "selu", "sigmoid", "tanh"],
    "dense3_dropout"    : [0, 0.2, 0.4, 0.5],

    "final_activation"  : ["sigmoid", "softmax", "tanh"],

    "optimizer"         : ["adam", "sgd", "adagrad"],
    "loss"              : ["binary_crossentropy", "mse", "mae", "kld", "msle"],
}

def random_select(trait):
    return trait[randint(0, len(trait)-1)]

class Genome:
    def __init__(self, mutation_rate=None, traits=None, create=True):
        # Percent of traits, roughly, that should be mutated
        if create is True:
            self.mutation_rate = 15
            self.traits = {}
            for t in traits_dict:
                self.traits[t] = random_select(traits_dict[t])
        else:
            self.mutation_rate = int(mutation_rate)
            self.traits = traits

    def breed(self, other_genome):
        child = Genome()
        for t in traits_dict:
            # By default, all traits are mutated. With a 100-mutation_rate chance, we inherit
            # at random from one of the two parents
            if randint(0, 100) > self.mutation_rate:
                child.traits[t] = random_select([self.traits[t], other_genome.traits[t]])
        return child


    def generate_model(self, params):
        model = Sequential()

        model.add(Conv2D(self.traits["conv1_neurons"], (3,2), input_shape=(2,3,params.window_sz * params.hz), padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation(self.traits["conv1_activation"]))
        model.add(MaxPooling2D(pool_size=(1, 2)))

        if self.traits["conv2_enabled"]:
            model.add(Conv2D(self.traits["conv2_neurons"], (3,2), padding='same'))
            model.add(BatchNormalization())
            model.add(Activation(self.traits["conv2_activation"]))
            model.add(MaxPooling2D(pool_size=(1,2)))

        if self.traits["conv3_enabled"]:
            model.add(Conv2D(self.traits["conv3_neurons"], (3,2), padding='same'))
            model.add(BatchNormalization())
            model.add(Activation(self.traits["conv3_activation"]))
            model.add(MaxPooling2D(pool_size=(1,2)))

        if self.traits["conv4_enabled"]:
            model.add(Conv2D(self.traits["conv4_neurons"], (3,2), padding='same'))
            model.add(BatchNormalization())
            model.add(Activation(self.traits["conv4_activation"]))
            model.add(MaxPooling2D(pool_size=(1,2)))

        if self.traits["conv5_enabled"]:
            model.add(Conv2D(self.traits["conv5_neurons"], (3,2), padding='same'))
            model.add(BatchNormalization())
            model.add(Activation(self.traits["conv5_activation"]))
            model.add(MaxPooling2D(pool_size=(1,2)))

        if self.traits["convdense_dropout"] != 0:
            model.add(Dropout(self.traits["convdense_dropout"]))

        model.add(Flatten())

        if self.traits["dense1_enabled"]:
            model.add(Dense(self.traits["dense1_neurons"]))
            model.add(BatchNormalization())
            model.add(Activation(self.traits["dense1_activation"]))
            model.add(Dropout(self.traits["dense1_dropout"]))

        if self.traits["dense2_enabled"]:
            model.add(Dense(self.traits["dense2_neurons"]))
            model.add(BatchNormalization())
            model.add(Activation(self.traits["dense2_activation"]))
            model.add(Dropout(self.traits["dense2_dropout"]))

        if self.traits["dense3_enabled"]:
            model.add(Dense(self.traits["dense3_neurons"]))
            model.add(BatchNormalization())
            model.add(Activation(self.traits["dense3_activation"]))
            model.add(Dropout(self.traits["dense3_dropout"]))


        model.add(Dense(1, kernel_initializer='normal'))
        model.add(BatchNormalization())
        model.add(Activation(self.traits["final_activation"]))

        model.compile(loss=self.traits["loss"],
                      optimizer=self.traits["optimizer"],
                      metrics=['accuracy'])
        return model



