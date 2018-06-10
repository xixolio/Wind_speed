# -*- coding: utf-8 -*-

from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras import optimizers, regularizers
import sys

''' Parameters format "[layer1-layer2-...-layern],lag,time_steps,epochs,l2,learning_rate" '''

def get_params():
    
    params = str(sys.argv[1])
    params = [*params.split(',')]
    
    layers = str(params[0]).strip('[]')
    layers = [int(layer) for layer in [*layers.split('-')]]
    
    lag = int(params[1])
    time_steps = int(params[2])
    epochs = int(params[3])
    l2 = int(params[4])
    learning_rate = int(params[5])
    
    return layers, lag, time_steps, epochs, l2, learning_rate
    



def model(layers, lag, time_steps, l2, learning_rate):
                     
    inputs = Input(batch_shape=(1, time_steps, lag))
    
    dummy_layer = inputs
    
    for layer in layers:
        
        lstm = LSTM(layer,return_sequences=False,activation='tanh',
                 recurrent_activation='sigmoid',dropout=0.0,recurrent_dropout=0.0,
                 activity_regularizer=regularizers.l2(l2),
                 recurrent_regularizer=regularizers.l2(l2),
                 stateful=True)
    
        dummy_layer = lstm(dummy_layer)
    
    outputs = Dense(1, activation = 'linear')(dummy_layer)

    model = Model(inputs = inputs, outputs = outputs)

    ad = optimizers.Adadelta(lr = learning_rate)
    
    model.compile(optimizer = ad, loss = 'mse')
    
    return model