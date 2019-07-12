from numpy import array
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
    checkpointPath = 'C:\Workspace\DataSource\MyLearning\TrainedModel\sentimentAnalysis.ckpt'
    epochNumber = 50
    repeatCount = 30
    tokenizer = Tokenizer()
    
    train_x, test_x, vocab = tp.getTrainAndTestData(vocab_filename, positiveSentenceDir, negativeSentenceDir, tokenizer)
    train_y = array([0 for _ in range(900)] + [1 for _ in range(900)])
    test_y = array([0 for _ in range(100)] + [1 for _ in range(100)])
    
    demoModel = sentiment.createSentimentAnalysisModel(test_x.shape[1]);
    #scores = sentiment.trainSentimentAnalysisModel(demoModel, train_x, train_y, test_x, test_y, epochNumber, checkpointPath, repeatCount)
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

