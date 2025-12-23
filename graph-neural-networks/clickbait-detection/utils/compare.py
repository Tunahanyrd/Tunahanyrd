from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def compare_bayes(df):
    X_train, X_test, y_train, y_test = train_test_split(df["title"], df["fake"], test_size=0.2, random_state=42)
    pipe = make_pipeline(CountVectorizer(), MultinomialNB())
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    from sklearn.metrics import ConfusionMatrixDisplay

    ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test)
    plt.title("Naive Bayes Confusion Matrix")
    plt.show()

    return y_test, y_pred

def compare_linearity(train_df, test_df, final_train_embeddings, final_test_embeddings):
    X_train = final_train_embeddings
    X_test  = final_test_embeddings
    y_train = train_df["fake"]
    y_test  = test_df["fake"]
    
    models = {
        "LogReg": LogisticRegression(max_iter=1000),
        "SVM": SVC(),
        "k-NN": KNeighborsClassifier(n_neighbors=5),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    }
    
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f"\n {name} Results:\n")
        print(classification_report(y_test, y_pred, digits=4))

def non_gat_train(train_df, test_df, train_emb, test_emb):
    X_train = train_emb
    X_test = test_emb
    y_train = train_df["fake"]
    y_test = test_df["fake"]
    
    models = {
        "LogReg": LogisticRegression(max_iter=1000),
        "SVM": SVC(),
        "k-NN": KNeighborsClassifier(n_neighbors=5),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    }
    
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f"\n {name} Results:\n")
        print(classification_report(y_test, y_pred, digits=4))













