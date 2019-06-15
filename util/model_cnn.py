from keras.applications import VGG19
from keras.applications import VGG16
from keras.applications import InceptionV3
from keras.applications import resnet50
from keras import models
from keras import layers
from keras import regularizers
from keras.preprocessing import image
import numpy as np
from sklearn.externals import joblib
import pickle
from keras.applications.vgg16 import preprocess_input

def modelVGG19():
    conv_base = VGG19(weights='imagenet', include_top=False, input_shape=(150,150,3))
    return conv_base

def modelVGG16():
    return VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))

def modelInceptionV3():
    return InceptionV3(weights='imagenet', include_top=False, input_shape=(150,150,3))

def modelResnet50():
    return resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

def modelLRResnet50():
    model = models.Sequential()
    conv_base = modelResnet50()
    model.add(conv_base)
    return model

def modelSimple():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    #model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

def modelSimple1():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

def modelSimple2():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    return model

def modelSimple3():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

def modelSimple4():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    model.add(layers.Dense(10, activation='softmax'))
    return model

def modelSimple412():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    model.add(layers.Dense(12, activation='softmax'))
    return model

def modelSimple4_fruit360():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    model.add(layers.Dense(75, activation='softmax'))
    return model

def modelSimpleGhim20k():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    model.add(layers.Dense(20, activation='softmax'))
    return model

def modelSimple5():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(150, 150, 3)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (5, 5), activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    model.add(layers.Conv2D(32, (5, 5), activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    model.add(layers.Dense(10, activation='softmax'))
    return model


def modelSimple41():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (5, 5), activation='relu',
                            input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (5, 5), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (5, 5), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (5, 5), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.Dense(10, activation='softmax'))
    return model


def modelSimple41_general(sum_classes=10):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (5, 5), activation='relu',
                            input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (5, 5), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (5, 5), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (5, 5), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.Dense(sum_classes, activation='softmax'))
    return model

def modelVGG19Mod(sum_classes=10):
    model = models.Sequential()
    conv_base = modelVGG19()
    model.add(conv_base)
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(sum_classes, activation='softmax'))
    return model

def modelVGG16Mod(sum_classes=10):
    model = models.Sequential()
    conv_base = modelVGG16()
    model.add(conv_base)
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(sum_classes, activation='softmax'))
    return model

def modelRestNetMod(sum_classes=10):
    model = models.Sequential()
    conv_base = modelResnet50()
    #conv_base.summary()
    model.add(conv_base)
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(sum_classes, activation='softmax'))
    return model

def save_model_joblib(model, file_name_model = "cnn_model.joblib.dat"):
    # save model to file
    joblib.dump(model, file_name_model)

def predict_by_model_joblib(x_test, file_name_model="cnn_model.joblib.dat"):
    # load model from file
    loaded_model = joblib.load(file_name_model)
    # make predictions for test data
    y_pred = loaded_model.predict(x_test)
    predictions = [round(value) for value in y_pred]
    return predictions


def save_model_pickle(model, file_name_model = "cnn_model.joblib.dat"):
    # save model to file
    pickle.dump(model, open(file_name_model, "wb"))

def predict_by_model_pickle(x_test, file_name_model="cnn_model.joblib.dat"):
    # load model from file
    loaded_model = pickle.load(open(file_name_model, "rb"))
    # make predictions for test data
    y_pred = loaded_model.predict(x_test)
    predictions = [round(value) for value in y_pred]
    return predictions

# mengambil banyak index layer activation cnn
def getIndexActivationLayer(model):
    return len(model.layers)-1

# mengambil banyaknya layer activation cnn
def getSumActivationLayer(model):
    return len(model.layers)

# mengembailkan nama-nama layer yang didefiniskan pada model
def getNamesLayer(model):
    return [layer.name for layer in model.layers[:]]

# mengembialikan nama layer pada index tertentu, dengan default index = 0
def getNameThe_Layer(model, index=0):
    lsname = [layer.name for layer in model.layers]
    n = getSumActivationLayer(model)
    # untuk index pertama dari depan berarti index = -1, index kedua dari depan berarti index = -2
    if index >= n or index <= -n:
        return
    else:
        return lsname[index]

# mengembalikan list output-output layer yang dimiliki oleh model
def getOutputLayer(model):
    return [layer.output for layer in model.layers]

def getOutputThe_Layer(model, index=0):
    lsoutput = [layer.output for layer in model.layers]
    n = getSumActivationLayer(model)
    # untuk index pertama dari depan berarti index = -1, index kedua dari depan berarti index = -2
    if index >= n or index <= -n:
        return
    else:
        return lsoutput[index]

# mengembalikan nama output berdasarkan layer_name yang diberikan
def getOutputThe_LayerByName(model, layer_name):
    return model.get_layer(layer_name).output

def getActivationImage(activation_model, img_tensor):
    return activation_model.predict(img_tensor)

def getModelInput(model):
    return model.input

def getNameModelInput(model):
    return model.input.name

# mengembalikan shape dari model input
def getShapeModelInput(model):
    return model.input.shape

# mengembalikan model output
def getModelOutput(model):
    return model.output

# mengembalikan name model output cnn
def getNameModelOutput(model):
    return model.output.name

# mengembalikan shape output model cnn
def getShapeModelOutput(model):
    return model.output.shape

# mengembalikan semua activation model
def getAllActivation(model, img_tensor):
    activation_model = models.Model(inputs=model.input, outputs=getOutputLayer(model))
    activation = activation_model.predict(img_tensor)
    return activation

# mengembalikan semua activation model cnn dalam bentuk array
def getListActivation(model, img_tensor):
    activation_model = models.Model(inputs=model.input, outputs=getOutputLayer(model))
    activation = activation_model.predict(img_tensor)
    return [act for act in (activation)]

# penulisan lain dengan method getThe_Activation(model, img_tensor, index)
def getElemenThe_Activation(model, im_tensor, index= 0):
    n = getSumActivationLayer(model)
    if index >= n or index <= -n:
        return
    else:
        return getListActivation(model, img_tensor=im_tensor)[index]

# penulisan lain dengan method getElemenThe_Activation(model, img_tensor, index)
# peng-index-an dimulai dari 0
def getThe_Activation(model, img_tensor, index=0):
    activation_model = models.Model(inputs=model.input, outputs=getOutputLayer(model))
    n = getSumActivationLayer(model)
    if index>= n or index <= -n:
        return
    else:
        return activation_model.predict(img_tensor)[index]

# mengembalikan feature activation model ke-i
def getFeatureThe_Activation(model, img_tensor, index=0):
    activation_model = models.Model(inputs=model.input, outputs=getOutputLayer(model))
    n = getSumActivationLayer(model)
    if index >= n or index <= -n:
        return
    else:
        return activation_model.predict(img_tensor)[index].shape[-1] # karena index pertama adalah kernel

def getExtrationFeatureCNN(model, img_tensor):
    activation_model = models.Model(inputs=model.input, outputs=getOutputLayer(model))
    n = getSumActivationLayer(model)
    return activation_model.predict(img_tensor)[n-2]

def getExtrationFeatureCNNBeforeOut(model, img_tensor, index=1):
    activation_model = models.Model(inputs=model.input, outputs=getOutputLayer(model))
    n = getIndexActivationLayer(model)
    return activation_model.predict(img_tensor)[n-index]

# mengembalikan matriks feature dari citra
def getEFCNN(model, data_train):
    #print("Get EF CNN")
    activation_model = models.Model(inputs=model.input, outputs=getOutputLayer(model))
    n = getIndexActivationLayer(model)

    img_tensor = np.expand_dims(data_train[0], axis=0)
    fcnn = activation_model.predict(img_tensor)[n-1]
    print(data_train.shape[0])
    print(data_train.shape)
    #print(fcnn)
    ls_feature = np.zeros((data_train.shape[0], fcnn.shape[1]))
    ls_feature[0] = fcnn

    for x in range(1, data_train.shape[0]):
        img_tensor = np.expand_dims(data_train[x], axis=0)
        fcnn = activation_model.predict(img_tensor)[n-1]
        ls_feature[x] = fcnn

        if 0==(x%50):
            print("Sudah mengekstrak data sebanyak ", x, "!")

    return ls_feature


# mengembalikan matriks feature dari citra
def getEFCNNPretrained(model, data_train):
    # print("Get EF CNN")
    n = getIndexActivationLayer(model)
    print("Banyaknya n=", n)
    img_tensor = np.expand_dims(data_train[0], axis=0)
    img_data = preprocess_input(img_tensor)
    fcnn = model.predict(img_tensor)
    print(fcnn.shape)
    ls_feature = np.zeros((data_train.shape[0], fcnn.shape[1]))
    ls_feature[0] = fcnn

    for x in range(1, data_train.shape[0]):
        img_tensor = np.expand_dims(data_train[x], axis=0)
        fcnn = model.predict(img_tensor)
        ls_feature[x] = fcnn

        if 0 == (x % 50):
            print("Sudah mengekstrak data sebanyak ", x, "!")

    return ls_feature

# mengembalikan matriks feature dari citra
def getEFCNNByLayerBeforeTop(model, data_train, my_layer_before_top = 1):
    #print("Get EF CNN")
    activation_model = models.Model(inputs=model.input, outputs=getOutputLayer(model))
    n = getIndexActivationLayer(model)
    img_tensor = np.expand_dims(data_train[0], axis=0)
    fcnn = activation_model.predict(img_tensor)[n-my_layer_before_top]
    print(data_train.shape[0])
    print(data_train.shape)
    print("Banyaknya n=", n)
    #print(fcnn)
    ls_feature = np.zeros((data_train.shape[0], fcnn.shape[1]))
    ls_feature[0] = fcnn

    for x in range(1, data_train.shape[0]):
        img_tensor = np.expand_dims(data_train[x], axis=0)
        fcnn = activation_model.predict(img_tensor)[n-my_layer_before_top]

        ls_feature[x] = fcnn

        if 0==(x%50):
            print("Sudah mengekstrak data sebanyak ", x, "!")

    return ls_feature

