from sklearn.model_selection import train_test_split
from data import load_data, preprocess, create_sequences
from model import build_model
from visualize import plot_results

def run(path):
    df = load_data(path)
    df = preprocess(df)

    X, y = create_sequences(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = build_model()

    model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.1)

    pred = (model.predict(X_test) > 0.5).astype(int)

    from sklearn.metrics import confusion_matrix, classification_report
    history = model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.1)
    plot_results(y_test, pred, history)

    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))

if __name__ == "__main__":
    run("D:/curr/ukdale.h5")