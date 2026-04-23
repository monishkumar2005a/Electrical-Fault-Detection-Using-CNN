import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_results(y_test, pred, history):
    cm = confusion_matrix(y_test, pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig("images/confusion_matrix.png")
    plt.close()

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['Train','Validation'])
    plt.title("Loss Curve")

    plt.savefig("images/loss_curve.png")
    plt.close()