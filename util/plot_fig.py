import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import itertools
from keras.preprocessing.image import load_img, img_to_array, image
from keras.utils.vis_utils import plot_model


def display_image(img_path):
    img = load_img(img_path, target_size=(150, 150))
    img_tensor = img_to_array(img)  # menghasilkan dimensi (150, 150, 3)
    #print("Shape image= ", img_tensor.shape)
    img_tensor = np.expand_dims(img_tensor, axis=0)  # (1, 150, 150, 3)
    #print("Shape setelah expand= ", img_tensor.shape)
    # Range nilai citra antara [0..1] untuk float [0 .. 255]
    img_tensor /= 255
    plt.imshow(img_tensor[0])
    plt.show()


def plot_history(history):
    fig = plt.figure()
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

    ## As loss always exists
    epochs = range(1, len(history.history[loss_list[0]]) + 1)

    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], color='blue', linestyle='--', linewidth=2, marker='o', markersize=5,
                 label='Training loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], color='green', linestyle='-', linewidth=2, marker='*', markersize=5,
                 label='Validation loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))

    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    fig.savefig("img/loss-history.jpg")
    #plt.close(fig)

    ## Accuracy
    #plt.figure(2)
    fig2 = plt.figure()
    for l in acc_list:
        plt.plot(epochs, history.history[l], color='blue', linestyle='--', linewidth=2, marker='o', markersize=5,
                 label='Training accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], color='green', linestyle='-', linewidth=2, marker='*', markersize=5,
                 label='Validation accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')

    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    fig2.savefig("img/acc-history.jpg")
    #plt.close(fig2)

    plt.show()

def plot_history_by_dir_name(history, dir_name="img", file_name="flot"):
    fig = plt.figure()
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

    ## As loss always exists
    epochs = range(1, len(history.history[loss_list[0]]) + 1)

    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], color='blue', linestyle='--', linewidth=2, marker='o', markersize=5,
                 label='Training loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], color='green', linestyle='-', linewidth=2, marker='*', markersize=5,
                 label='Validation loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))

    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    fig.savefig(dir_name + str('/') + file_name + str('_loss_') + ".jpg")
    #plt.close(fig)

    ## Accuracy
    #plt.figure(2)
    fig2 = plt.figure()
    for l in acc_list:
        plt.plot(epochs, history.history[l], color='blue', linestyle='--', linewidth=2, marker='o', markersize=5,
                 label='Training accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], color='green', linestyle='-', linewidth=2, marker='*', markersize=5,
                 label='Validation accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')

    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    fig2.savefig(dir_name + str('/') + file_name + str('_acc_')+".jpg")
    #plt.close(fig2)
    # jika tidang ingin ditampilkan kepada pengguna
    #plt.show()

def plot_history_save_img(history, savefilename="img/model"):
    fig = plt.figure()
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

        ## As loss always exists
    epochs = range(1, len(history.history[loss_list[0]]) + 1)

    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    fig.savefig(savefilename+str('_loss_')+".jpg")
    plt.close(fig)

    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    fig.savefig(savefilename+str('_acc_')+".jpg")
    plt.close(fig)

def plot_fig(i, history, epochs):
    fig = plt.figure()
    plt.plot(range(1,epochs+1),history.history['val_acc'],label='validation')
    plt.plot(range(1,epochs+1),history.history['acc'],label='training')
    plt.legend(loc=0)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.xlim([1,epochs])
    # plt.ylim([0,1])
    plt.grid(True)
    plt.title("Model Accuracy")
    plt.show()
    fig.savefig('img/'+str(i)+'-accuracy.jpg')
    plt.close(fig)


## multiclass or binary report
## If binary (sigmoid output), set binary parameter to True
def full_multiclass_report(model,
                           x,
                           y_true,
                           classes,
                           batch_size=32,
                           binary=False):
    # 1. Transform one-hot encoded y_true into their class number
    if not binary:
        y_true = np.argmax(y_true, axis=1)

    # 2. Predict classes and stores in y_pred
    y_pred = model.predict_classes(x, batch_size=batch_size)

    # 3. Print accuracy score
    #print("Accuracy : " + str(accuracy_score(y_true, y_pred)))

    #print("")

    # 4. Print classification report
    #print("Classification Report")
    print(classification_report(y_true, y_pred, digits=5))

    # 5. Plot confusion matrix
    cnf_matrix = confusion_matrix(y_true, y_pred)
    #print(cnf_matrix)
    plot_confusion_matrix(cnf_matrix, classes=classes)



def plot_confusion_matrix_by_predict(y_true, y_pred, classes, savefilename="img/conf_matrix.jpg"):
    print(classification_report(y_true, y_pred, digits=5))
    # 5. Plot confusion matrix
    cnf_matrix = confusion_matrix(y_true, y_pred)
    #print(cnf_matrix)
    plot_confusion_matrix_save_img(cnf_matrix, classes=classes, savefilename=savefilename)


def full_multiclass_report_save_img(model,
                           x,
                           y_true,
                           classes,
                           batch_size=32,
                           binary=False, savefilename="img/cm.jpg"):
    # 1. Transform one-hot encoded y_true into their class number
    if not binary:
        y_true = np.argmax(y_true, axis=1)

    # 2. Predict classes and stores in y_pred
    y_pred = model.predict_classes(x, batch_size=batch_size)

    # 3. Print accuracy score
    print("Accuracy : " + str(accuracy_score(y_true, y_pred)))

    #print("")

    # 4. Print classification report
    #print("Classification Report")
    print(classification_report(y_true, y_pred, digits=5))

    # 5. Plot confusion matrix
    cnf_matrix = confusion_matrix(y_true, y_pred)
    #print(cnf_matrix)
    plot_confusion_matrix_save_img(cnf_matrix, classes=classes, savefilename=savefilename)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized confusion matrix'
    else:
        title = 'Confusion matrix'

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_confusion_matrix_save_img(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues, savefilename="img/cm.jpg"):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized confusion matrix'
    else:
        title = 'Confusion matrix'

    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    fig.savefig(savefilename)
    plt.close(fig)

def plot_figure_acc(epochs, history, xLabel="Epochs", yLabel= "Accuracy", title="Model Training and validation accuration", savefilename="img/-accuracy.jpg"):
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    value_max_acc = max(acc_list)
    print(acc_list)
    print("Nilai max= ", value_max_acc)

    fig = plt.figure()
    plt.plot(range(1, epochs + 1), history.history['acc'], 'b', label='Training acc')
    plt.plot(range(1, epochs + 1), history.history['val_acc'],'g', label='Validation acc')
    plt.legend(loc=0)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.xlim([1, epochs])
    plt.ylim([0, 1])
    plt.grid(True)
    plt.title(title)
    plt.show()
    fig.savefig(savefilename)
    plt.close(fig)

def plot_figure_loss(epochs, history, xLabel="Epochs", yLabel= "Loss", title="Model Training and validation loss", savefilename="img/-loss.jpg"):
    fig = plt.figure()
    plt.plot(range(1, epochs + 1), history.history['loss'], label='Training loss')
    plt.plot(range(1, epochs + 1), history.history['val_loss'], label='Validation loss')
    plt.legend(loc=0)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.xlim([1, epochs])
    plt.ylim([0, 1])
    plt.grid(True)
    plt.title(title)
    plt.show()
    fig.savefig(savefilename)
    plt.close(fig)


def plot_figure_acc_loss(epochs, history, xLabel="Epochs", yLabel= "Loss/Accuracy", title="Model Training and validation acc-loss", savefilename="img/-acc-loss.jpg"):
    fig = plt.figure()
    plt.plot(range(1, epochs + 1), history.history['acc'], label='Training acc')
    plt.plot(range(1, epochs + 1), history.history['val_acc'], label='Validation acc')
    plt.plot(range(1, epochs + 1), history.history['loss'], label='Training loss')
    plt.plot(range(1, epochs + 1), history.history['val_loss'], label='Validation loss')
    plt.legend(loc=0)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.xlim([1, epochs])
    plt.ylim([0, 1])
    plt.grid(True)
    plt.title(title)
    plt.show()
    fig.savefig(savefilename)
    plt.close(fig)

def plot_multiple_figure_all(epochs, ls_history, ls_label, xLabel="Epochs", yLabel= "Accuracy", title="Model Accuracy", savefilename="img/-accuracy.jpg"):
    fig = plt.figure(figsize=(12, len(ls_history)))

    for x in range (0, len(ls_history)):
        plt.plot(range(epochs), ls_history[x].history['acc'], label="Training acc " +ls_label[x])
        plt.plot(range(epochs), ls_history[x].history['val_acc'], label="Validation acc "+ ls_label[x])
        plt.plot(range(epochs), ls_history[x].history['loss'], label="Training loss " +ls_label[x])
        plt.plot(range(epochs), ls_history[x].history['val_loss'], label="Validation loss " + ls_label[x])

    plt.legend(loc=0)
    plt.xlabel(xLabel)
    plt.xlim([0, epochs])
    plt.ylabel(yLabel)
    plt.grid(True)
    plt.title(title)
    plt.show()
    fig.savefig(savefilename)
    plt.close(fig)

def plot_multiple_figure_train_acc(epochs, ls_history, ls_label, xLabel="Epochs", yLabel= "Accuracy", title="Model Training Accuracy", savefilename="img/train-accuracy.jpg"):
    fig = plt.figure(figsize=(12, len(ls_history)))

    for x in range (0, len(ls_history)):
        plt.plot(range(epochs), ls_history[x].history['acc'], label=ls_label[x])

    plt.legend(loc=0)
    plt.xlabel(xLabel)
    plt.xlim([0, epochs])
    plt.ylabel(yLabel)
    plt.grid(True)
    plt.title(title)
    plt.show()
    fig.savefig(savefilename)
    plt.close(fig)

def plot_multiple_figure_val_acc(epochs, ls_history, ls_label, xLabel="Epochs", yLabel= "Accuracy", title="Model Validation Accuracy", savefilename="img/val-accuracy.jpg"):
    fig = plt.figure(figsize=(12, len(ls_history)))

    for x in range (0, len(ls_history)):
        plt.plot(range(epochs), ls_history[x].history['val_acc'], label=ls_label[x])


    plt.legend(loc=0)
    plt.xlabel(xLabel)
    plt.xlim([0, epochs])
    plt.ylabel(yLabel)
    plt.grid(True)
    plt.title(title)
    plt.show()
    fig.savefig(savefilename)
    plt.close(fig)

def plot_multiple_figure_train_loss(epochs, ls_history, ls_label, xLabel="Epochs", yLabel= "Loss", title="Model Training Loss", savefilename="img/train-loss.jpg"):
    fig = plt.figure(figsize=(12, len(ls_history)))

    for x in range (0, len(ls_history)):
        plt.plot(range(epochs), ls_history[x].history['loss'], label=ls_label[x])


    plt.legend(loc=0)
    plt.xlabel(xLabel)
    plt.xlim([0, epochs])
    plt.ylabel(yLabel)
    plt.grid(True)
    plt.title(title)
    plt.show()
    fig.savefig(savefilename)
    plt.close(fig)

def plot_multiple_figure_val_loss(epochs, ls_history, ls_label, xLabel="Epochs", yLabel= "Loss", title="Model Validation Accuracy", savefilename="img/val-loss.jpg"):
    fig = plt.figure(figsize=(12, len(ls_history)))

    for x in range (0, len(ls_history)):
        plt.plot(range(epochs), ls_history[x].history['val_loss'], label=ls_label[x])

    plt.legend(loc=0)
    plt.xlabel(xLabel)
    plt.xlim([0, epochs])
    plt.ylabel(yLabel)
    plt.grid(True)
    plt.title(title)
    plt.show()
    fig.savefig(savefilename)
    plt.close(fig)

def getImageTensor(image_path='./image.jpg', target_size=(150,150)):
    img = load_img(image_path, target_size=target_size)
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0) # disisipkan searah sumbu x, axis=1 searah sumbu y
    img_tensor /=255
    return img_tensor

def showImage(image_path='./image.jpg', target_size=(150,150)):
    img = load_img(image_path, target_size=target_size)
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0) # disisipkan searah sumbu x, axis=1 searah sumbu y
    img_tensor /=255
    plt.imshow(img_tensor[0])
    plt.show()
    plt.close()

def plotVisualConvThe_(layer_activation, index = 0):
    # jumlah feature dalam feature map, karena banyaknya feature yang direpresentasikan tergantung pada jumlah kernel yang dibentuk
    # (150, 150, 3) ini berarti ada 3 feature
    plt.matshow(layer_activation[0, :, :, index], cmap='viridis')
    plt.show()

def plotVisualConv(layer_activation, images_per_row = 16, layer_name="Conv_0"):
    # jumlah feature dalam feature map, karena banyaknya feature yang direpresentasikan tergantung pada jumlah kernel yang dibentuk
    # (150, 150, 3) ini berarti ada 3 feature
    n_feature = layer_activation.shape[-1]
    # feature map memiliki shape (I, size, size, n_feature)
    size = layer_activation.shape[1]
    n_cols = n_feature // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    np.seterr(divide='ignore', invalid='ignore')

    for col in range(n_cols):
        for row in range(images_per_row):
            # print("JUmlah = ", (col * images_per_row) + row)
            channel_image = layer_activation[0, :, :, (col * images_per_row) + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 32
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))

    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()

def plotVisualAllConv(layer_activations, images_per_row = 16, layer_names="Conv_0"):
    # display the feature maps
    for layer_name, layer_activation in zip(layer_names, layer_activations):
        plotVisualConv(layer_activation, layer_name=layer_name, images_per_row=images_per_row)

def plot_model_cnn(model, filename="model_plot.jpg"):
    plot_model(model, to_file=filename, show_shapes=True, show_layer_names=True)
