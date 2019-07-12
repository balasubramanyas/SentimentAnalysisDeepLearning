import com.bala.poc.textprocess.TextProcess as tp;

def predict_sentiment(review, vocab, tokenizer, model):
    # clean
    tokens = tp.clean_doc(review)
    # filter by vocab
    tokens = [w for w in tokens if w in vocab]
    # convert to line
    line = ' '.join(tokens)
    # encode
    encoded = tokenizer.texts_to_matrix([line], mode='freq')
    # prediction
    yhat = model.predict(encoded, verbose=0)
    return round(yhat[0,0])