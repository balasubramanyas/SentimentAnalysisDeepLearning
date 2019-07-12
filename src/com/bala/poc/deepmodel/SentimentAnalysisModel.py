import tensorflow.keras as tfk;
import numpy as np;
import tensorflow as tf;

def createSentimentAnalysisModel(inputshape):

    model = tfk.models.Sequential()
    model.add(tfk.layers.Dense(50, input_shape=(inputshape,), activation='relu'))
    model.add(tfk.layers.Dense(1, activation='sigmoid'))
    # compile network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
    return model

def trainSentimentAnalysisModel(model, train_x, train_y, test_x, test_y, epochNumber, checkpoint_path, repeatCount) :

    scores = list()
    
    # Create checkpoint callback
    cp_callback = tfk.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=2)
    #for i in range(repeatCount):
    model.fit(train_x,train_y, epochs=epochNumber, verbose=2, callbacks = [cp_callback])
    loss, acc = model.evaluate(test_x, test_y, verbose=2)
    scores.append(acc)
    return scores;
    

def createLstmModel(X):
    # define model
    model = tfk.models.Sequential()
    model.add(tfk.layers.LSTM(75, input_shape=(X.shape[1], X.shape[2])))
    model.add(tfk.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    
    return model;

def trainLSTMModel(model, train_x, epochNumber, checkpoint_path) :

    scores = list()
    
    # Create checkpoint callback
    cp_callback = tfk.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=2)
    #for i in range(repeatCount):
    model.fit(np.array(train_x), epochs=epochNumber, verbose=2, callbacks = [cp_callback])
    loss, acc = model.evaluate(np.array(train_x), verbose=2)
    scores.append(acc)
    return scores;
        
    

