from sklearn.metrics import confusion_matrix, classification_report

def evaluate(model, X_test, y_test):
    pred = (model.predict(X_test) > 0.5).astype(int)

    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))