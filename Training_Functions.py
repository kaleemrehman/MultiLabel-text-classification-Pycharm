import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

############# Function for text processing ##########################
def text_process(mess):
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


############ Function to train model on connection issues ################
def Connection_Training(df):
    msg_train, msg_test, label_train, label_test = \
        train_test_split(df['review'], df['Bluetooth'], test_size=0.2)
    pipeline = Pipeline([
        ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
        ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
        ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
    ])
    pipeline.fit(msg_train, label_train)
    predictions = pipeline.predict(msg_test)
    print(classification_report(predictions, label_test))
    return pipeline


############ Function to train model on VcSetting (status) issues ################
def Settings_Training(df2):
    msg_train, msg_test, label_train, label_test = \
        train_test_split(df2['review'], df2['VC-Settings'], test_size=0.2)
    pipeline = Pipeline([
        ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
        ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
        ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
    ])
    pipeline.fit(msg_train, label_train)
    predictions = pipeline.predict(msg_test)
    print(classification_report(predictions, label_test))
    return pipeline


############ Function to train model on Volume issues ################
def Volume_Training(df3):
    msg_train, msg_test, label_train, label_test = \
        train_test_split(df3['review'], df3['Volume'], test_size=0.2)
    pipeline = Pipeline([
        ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
        ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
        ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
    ])
    pipeline.fit(msg_train, label_train)
    predictions = pipeline.predict(msg_test)
    print(classification_report(predictions, label_test))
    return pipeline
