from util.ImageGenerator import generateData
import numpy as np

class BuildDataset(object):

    def __init__(self, dir_name='./', input_shape=(150, 150, 3), batch_size=10, counter=1, class_mode='categorical'):
        self.__dirname = dir_name
        self.__input_shape = input_shape
        self.__batch_size = batch_size
        self.__counter = counter
        self.__class_mode = class_mode

    def setDirName(self, dir_name):
        self.__dirname = dir_name

    def getDirName(self):
        return self.__dirname

    def setCategory(self, category):
        self.__category = category

    def getCategory(self):
        return self.__category

    def setLabels(self, labels):
        self.__labels = labels

    def getLabels(self):
        return self.__labels

    def setDataset(self, dataset):
        self.__dataset = dataset

    def getDataset(self):
        return self.__dataset

    def setInputShape(self, input_shape):
        self.__input_shape = input_shape

    def getInputShape(self):
        return self.__input_shape


    def generateDataset(self):

        # image_width, image_height, channels = (130, 130, 3)
        # Jumlah dataset yang dibangkitkan dari direktori citra harus lebih kecil atau sama dengan jumlah citra yang ada
        # pada direktori citra
        input_shape = self.getInputShape()
        train_generator = generateData(train_dir= self.getDirName(), input_shape=(input_shape[0], input_shape[1]),
                                       batch_size=self.__batch_size,
                                       class_mode=self.__class_mode)
        class_dictionary = train_generator.class_indices

        # print(class_dictionary)

        # countCat = getAllCategoryDir(dir)
        # print("Banyaknya kategory: ", len(countCat))
        # print(countCat)
        dataset = np.ndarray(shape=(self.__counter * self.__batch_size, input_shape[0], input_shape[1], input_shape[2]),
                             dtype=np.float32)
        labels = np.ndarray(shape=(self.__counter * self.__batch_size, len(class_dictionary)), dtype=np.int8)

        i = 0
        start = i
        end = self.__batch_size
        for data_batch, labels_batch in train_generator:
            # print(labels_batch)
            # print('Data batch shape: ', data_batch.shape)
            # print('Labels batch sahpe: ', labels_batch.shape)
            # print("Awal = %d akhir = %d" % (start, end))
            # untuk menset nilai elemen dataset mulai dari elemen start sampai elemen end
            # dengan nilai data_batch yang dibandingkan
            dataset[start:end] = data_batch
            labels[start:end] = labels_batch
            start = end
            i += 1
            end = (i + 1) * self.__batch_size  # karena dimulai dari 0

            if i % self.__counter == 0:
                break

        self.setLabels(labels)
        self.setCategory(class_dictionary)
        self.setDataset(dataset)

        return (dataset, labels, class_dictionary)
