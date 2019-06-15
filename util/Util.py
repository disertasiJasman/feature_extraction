# Require Python Package
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score

# pada perhitungan scores dihitung berdasarkan label_test categorical, misal untuk african [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
def getAccuracy(model, data_test, label_test):
    print("Accuracy")
    scores = model.evaluate(data_test, label_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

# pada perhitungan accuracy_score dihitung berdasarkan label_target biner bukan categorical, misal untuk african 0
def getAccuracyByPredict(model, data_test, label_test):
    lsPredic = model.predict_classes(data_test)
    accuracy = accuracy_score(label_test, lsPredic)
    print("Accuracy Model Wang6 : %.2f%%" % (accuracy * 100.0))

def sigmoid(inputs):
    """
        Calculate the sigmoid for the give inputs (array)
        :param inputs:
        :return:
    """
    sigmoid_scores = [1 / float(1 + np.exp(- x)) for x in inputs]
    return sigmoid_scores


def softmax(inputs):
    """
    Calculate the softmax for the give inputs (array)
    :param inputs:
    :return:
    """
    return np.exp(inputs) / float(sum(np.exp(inputs)))


def line_graph(x, y, x_title, y_title):
    """
    Draw line graph with x and y values
    :param x:
    :param y:
    :param x_title:
    :param y_title:
    :return:
    """
    plt.plot(x, y)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.show()

def getImageDataGenerator():
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    return datagen
def getTestImageDataGenerator():
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    return test_datagen

def main():
    """
    graph_x = range(0, 21)
    graph_y = sigmoid(graph_x)

    print("Graph X readings: {}".format(graph_x))
    print("Graph Y readings: {}".format(graph_y))

    line_graph(graph_x, graph_y, "Inputs", "Sigmoid Scores")

    sigmoid_inputs = [2, 3, 5, 6, -3]
    print("Sigmoid Function Output :: {}".format(sigmoid(sigmoid_inputs)))
"""
    softmax_inputs = [2, 3, 5, 6]
    print("Softmax Function Output :: {}".format(softmax(softmax_inputs)))

    graph_x = range(0, 21)
    graph_y = softmax(graph_x)

    print("Graph X readings: {}".format(graph_x))
    print("Graph Y readings: {}".format(graph_y))

    line_graph(graph_x, graph_y, "Inputs", "Softmax Scores")

if __name__ =="__main__":
    main()


