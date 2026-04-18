from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

def train_nb(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train):
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    return model
