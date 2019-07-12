from string import punctuation
from os import listdir
from nltk.corpus import stopwords

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def clean_doc(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

def doc_to_line(filename, vocab):
    doc = load_doc(filename)
    tokens = clean_doc(doc)
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)

def process_docs(directory, vocab, is_trian):
    lines = list()
    for filename in listdir(directory):
        if is_trian and filename.startswith('cv9'):
            continue
        if not is_trian and not filename.startswith('cv9'):
            continue
        path = directory + '/' + filename
        line = doc_to_line(path, vocab)
        lines.append(line)
    return lines

def getTrainAndTestData(vocab_filename, positiveSentenceDir, negativeSentenceDir, tokenizer):
    vocab =load_doc(vocab_filename)
    vocab = vocab.split()
    vocab = set(vocab)
    
    positive_lines = process_docs(positiveSentenceDir, vocab, True)
    negative_lines = process_docs(negativeSentenceDir, vocab, True)

    docs = negative_lines + positive_lines
    tokenizer.fit_on_texts(docs)
    
    Xtrain = tokenizer.texts_to_matrix(docs, mode='freq')
    
    positive_lines = process_docs(positiveSentenceDir, vocab, False)
    negative_lines = process_docs(negativeSentenceDir, vocab, False)
    docs = negative_lines + positive_lines
    
    Xtest = tokenizer.texts_to_matrix(docs, mode='freq')
    
    return Xtrain, Xtest, vocab;

def saveVocabulary(vocab, filename):
   
    min_occurane = 2
    tokens = [k for k,c in vocab.items() if c >= min_occurane]
    
    data = '\n'.join(tokens)
    file = open(filename, 'w')
    file.write(data)
    file.close()