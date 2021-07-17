from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import BatchNormalization
from os.path import dirname, isfile
import pathlib2 as pathlib
from keras.models import model_from_json


class Model:

    def get(params, topo_file, weights_file):
        if isfile(topo_file) and isfile(weights_file):
            print("model loaded from disk")
            return Model.load_model(topo_file, weights_file)
        else:
            return Model.build_model(params)

    def save_model(m, t, w):
        t_path = dirname(t)
        pathlib.Path(t_path).mkdir(parents=True, exist_ok=True)
        w_path = dirname(w)
        pathlib.Path(w_path).mkdir(parents=True, exist_ok=True)

        model_json = m.to_json()
        with open(t, "w") as json_file:
            json_file.write(model_json)
        m.save_weights(w)

    def _compile(m):
        m.compile(
            loss='mae',
            optimizer='adagrad',
            metrics=['accuracy']
        )

    def load_model(t, w):
        json_file = open(t, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights(w)
        Model._compile(model)
        return model

    # def build_model(params):
    #     model = Sequential()
    #     model.add(Conv2D(16, (3, 2), input_shape=(2, 3, params.window_sz * params.hz), padding='valid'))
    #     model.add(BatchNormalization())
    #     model.add(Activation('relu'))
    #     model.add(MaxPooling2D(pool_size=(1, 2)))
    #
    #     model.add(Conv2D(256, (3, 2), padding='same'))
    #     model.add(BatchNormalization())
    #     model.add(Activation('selu'))
    #     model.add(MaxPooling2D(pool_size=(1, 2)))
    #
    #     model.add(Conv2D(64, (3, 2), padding='same'))
    #     model.add(BatchNormalization())
    #     model.add(Activation('relu'))
    #     model.add(MaxPooling2D(pool_size=(1, 2)))
    #
    #     model.add(Conv2D(64, (3, 2), padding='same'))
    #     model.add(BatchNormalization())
    #     model.add(Activation('sigmoid'))
    #     model.add(MaxPooling2D(pool_size=(1, 2)))
    #
    #     model.add(Dropout(0.2))
    #
    #     model.add(Flatten())
    #
    #     model.add(Dense(128, kernel_initializer='normal'))
    #     model.add(BatchNormalization())
    #     model.add(Activation('tanh'))
    #     model.add(Dropout(0.4))
    #
    #     model.add(Dense(128, kernel_initializer='normal'))
    #     model.add(BatchNormalization())
    #     model.add(Activation('relu'))
    #     model.add(Dropout(0))
    #
    #     model.add(Dense(512, kernel_initializer='normal'))
    #     model.add(BatchNormalization())
    #     model.add(Activation('tanh'))
    #     model.add(Dropout(0.2))
    #
    #     model.add(Dense(1, kernel_initializer='normal'))
    #     model.add(BatchNormalization())
    #     model.add(Activation("tanh"))
    #
    #     Model._compile(model)
    #
    #     return model


    def build_model(params):
        model = Sequential()

        model.add(Conv2D(int(params.model["conv1_neurons"]), (3,2), input_shape=(2,3,params.window_sz * params.hz), padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation(params.model["conv1_activation"]))
        model.add(MaxPooling2D(pool_size=(1,2)))

        if params.model["conv2_enabled"]:
            model.add(Conv2D(params.model["conv2_neurons"], (3,2), padding='same'))
            model.add(BatchNormalization())
            model.add(Activation(params.model["conv2_activation"]))
            model.add(MaxPooling2D(pool_size=(1,2)))

        if params.model["conv3_enabled"]:
            model.add(Conv2D(params.model["conv3_neurons"], (3,2), padding='same'))
            model.add(BatchNormalization())
            model.add(Activation(params.model["conv3_activation"]))
            model.add(MaxPooling2D(pool_size=(1,2)))

        if params.model["conv4_enabled"]:
            model.add(Conv2D(params.model["conv4_neurons"], (3,2), padding='same'))
            model.add(BatchNormalization())
            model.add(Activation(params.model["conv4_activation"]))
            model.add(MaxPooling2D(pool_size=(1,2)))

        if params.model["conv5_enabled"]:
            model.add(Conv2D(params.model["conv5_neurons"], (3,2), padding='same'))
            model.add(BatchNormalization())
            model.add(Activation(params.model["conv5_activation"]))
            model.add(MaxPooling2D(pool_size=(1,2)))

        if params.model["convdense_dropout"] != 0:
            model.add(Dropout(params.model["convdense_dropout"]))

        model.add(Flatten())

        if params.model["dense1_enabled"]:
            model.add(Dense(params.model["dense1_neurons"]))
            model.add(BatchNormalization())
            model.add(Activation(params.model["dense1_activation"]))
            model.add(Dropout(params.model["dense1_dropout"]))

        if params.model["dense2_enabled"]:
            model.add(Dense(params.model["dense2_neurons"]))
            model.add(BatchNormalization())
            model.add(Activation(params.model["dense2_activation"]))
            model.add(Dropout(params.model["dense2_dropout"]))

        if params.model["dense3_enabled"]:
            model.add(Dense(params.model["dense3_neurons"]))
            model.add(BatchNormalization())
            model.add(Activation(params.model["dense3_activation"]))
            model.add(Dropout(params.model["dense3_dropout"]))


        model.add(Dense(1, kernel_initializer='normal'))
        model.add(BatchNormalization())
        model.add(Activation(params.model["final_activation"]))

        model.compile(loss=params.model["loss"],
                      optimizer=params.model["optimizer"],
                      metrics=['accuracy'])
        return model
