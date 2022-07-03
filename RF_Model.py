import neattext.functions as nfx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

############ TRAINING ######################
def Model(df2):
    y = df2[['Bluetooth', 'VC-Settings', 'Volume']]
    corpus = df2['review'].apply(nfx.remove_stopwords)
    tfidf = TfidfVectorizer()
    Xfeatures = tfidf.fit_transform(corpus).toarray()
    X_train, X_test, y_train, y_test = train_test_split(Xfeatures, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    fr_prediction = clf.predict(X_test)
    print(accuracy_score(y_test, fr_prediction))
    ######## saving model #####
    Random_Forest_clf_file = open("Random_Forest_clf_model_file.pkl", "wb")
    joblib.dump(clf, Random_Forest_clf_file)
    Random_Forest_clf_file.close()
    ############ Saving Vectorizer #####
    # Save Vectorizer
    RF_tfidf_vectorizer_file = open("RF_tfidf_vectorizer.pkl", "wb")
    joblib.dump(tfidf, RF_tfidf_vectorizer_file)
    RF_tfidf_vectorizer_file.close()

################# PREDICTION###########
def Predict(df2):
    model = joblib.load('Random_Forest_clf_model_file.pkl')
    tfidf_model = joblib.load('RF_tfidf_vectorizer.pkl')
    vec_example = tfidf_model.transform(df2['review'])
    pre = model.predict(vec_example)
    df2["Bluetooth"] = pre[:, 0]
    df2["VC-Settings"] = pre[:, 1]
    df2["Volume"] = pre[:, 2]
    return df2

