import numpy as np;
import os
import tensorflow as tf

from keras.preprocessing.text import Tokenizer;
import com.bala.poc.textprocess.TextProcess as tp;
import com.bala.poc.deepmodel.SentimentAnalysisModel as sentiment
import com.bala.poc.deepmodel.SentimentModelPredict as predictmodel

if __name__ == '__main__':
    vocab_filename = 'C:\Workspace\DataSource\MyLearning\Vocabulary.txt'
    positiveSentenceDir = 'C:\Workspace\DataSource\MyLearning\sentence_positive'
    negativeSentenceDir = 'C:\Workspace\DataSource\MyLearning\sentence_negative'
    checkpointPath = 'C:\Workspace\DataSource\MyLearning\LSTMModel\sentimentAnalysis.ckpt'
    epochNumber = 2
    repeatCount = 30
    tokenizer = Tokenizer()
    
    train_x, test_x, vocab = tp.getTrainAndTestData(vocab_filename, positiveSentenceDir, negativeSentenceDir, tokenizer)
   
    train_x = np.reshape(train_x, (train_x.shape[1], 1,  train_x.shape[0]))
    test_x = np.reshape(test_x, (test_x.shape[1], 1,  test_x.shape[0]))
    
    train_y = []
    train_y_temp = np.zeros((train_x.shape[1], 1,  train_x.shape[0]), dtype=int)
    for i in range (len(train_x)):
        train_y.insert(i, train_y_temp)
   
    test_y = []
    test_y_temp = np.zeros((test_x.shape[1], 1,  test_x.shape[0]), dtype=int)
    for i in range (len(test_x)):
        test_y.insert(i, test_y_temp)
        
        
    print(len(train_x))
    print(len(train_y))
    
    demoModel = sentiment.createLstmModel(train_x);
    scores = sentiment.trainLSTMModel(demoModel, train_x, epochNumber, checkpointPath)
    checkpoint_dir = os.path.dirname(checkpointPath)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    demoModel.load_weights(latest);
    
    print("Sentiment Analysis Model Summary :")
    print(demoModel.summary());
    lossValue, accuracyValue = demoModel.evaluate(test_x, test_y, verbose=2);
    accuracyValue = accuracyValue * 100;
    print("\n\nSentiment Analysis Model Accuracy : " + str(accuracyValue))
    
    # test negative text
    text = 'The movie has no content, waste in spending time'
    print("\n\n"+text)
    print(predictmodel.predict_sentiment(text, vocab, tokenizer, demoModel))
    # test positive text
    text = 'The movie was okay not great'
    print("\n\n"+text)
    print(predictmodel.predict_sentiment(text, vocab, tokenizer, demoModel))
    
    pass

