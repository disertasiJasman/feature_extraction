import numpy as np
import os
from keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
from PIL import Image
from util.UtilPath import getLastDirectoryPath
from util.UtilPath import getFileNameFromFilePath
from util.ImageGenerator import generateData

class BuildDataset(object):

    def __init__(self, dirname):
        self.__dirname = dirname

    def setDirName(self, dirname):
        self.__dirname = dirname

    def getDirName(self):
        return self.__dirname

    def setLabelsByDirName(self, dirname):
        self.__dirname = dirname
        self.__listLabels =[]

    def setLabels(self, labels):
        self.__listLabels = labels

    def getLabels(self):
        return self.__listLabels

    def setSumCategory(self, sumCategory):
        self.__sumCategory = sumCategory

    def getSumCategory(self):
        return self.__sumCategory

    def generateLabels(self):
        self.__labels = []
        self.__labels = [x for x in os.listdir(self.getDirName()) if not x.__contains__('.')]
        #print(self.__labels)
        self.setLabels(self.__labels)
        #return self.__labels

    def generateDictLabel(self):
        myDict = {}

        for myVal, myKey in enumerate(self.getLabels()):
            myDict[myKey] = myVal
        #print(myDict)
        self.setDictLabel(myDict)

    def setDictLabel(self, dictLabels):
        self.__dictLabels = dictLabels

    def getDictLabel(self):
        return self.__dictLabels

    def getKeysByValue(self, valueToFind):
        listOfKeys = list()
        listOfItems = self.getDictLabel().items()
        for item in listOfItems:
            if item[1] == valueToFind:
                return item[0]
                break

    def getListOfFiles(self):
        listOfFiles = list()
        listOfFiles = [os.path.join(dp, f) for dp, dn, filenames in os.walk(self.getDirName()) for f in filenames if os.path.splitext(f)[1] == '.jpg']

        """
        for root, dirs, files in os.walk(self.getDirName(), topdown=False):
            listOfFiles += [os.path.join(root, file) for file in files]
        """
        return listOfFiles

    def getCategory(self, filename):
        # untuk mengamnbil nama direktori akhir sebelum nama file
        category = getLastDirectoryPath(filename)
        # mengambil dictionary category image
        myDict = self.getDictLabel()

        if category in myDict.keys():
            return myDict.get(category)
        else:
            print("Not contains in dictionary")

    # membangkitkan dataset secara manual
    def buildDataset(self):
        listAllImages = self.getListOfFiles()

        # Original Dimensions
        image_width = 130
        image_height = 130

        channels = 3
        dataset = np.ndarray(shape=(len(listAllImages), image_width, image_height, channels),
                             dtype=np.float32)
        # array 2 dimensi dengan baris jumlah image dataset, dan kolomnya 2
        target = np.ones(shape=(len(listAllImages), 1), dtype=np.int8)

        #print("Build Dataset")
        #print(len(listAllImages))
        #display(_Imgdis(self.getListOfFiles()[1], width=240, height=320))
        #print("Shape dataset: ", dataset.shape)
        #data = np.ndarray((image_width, image_height, channels),dtype=np.float32)

        i=0
        for file in listAllImages:
            #print(os.path.basename(os.path.dirname(dir)))
            img = load_img(file)  # this is a PIL image
            #print("Nama file: ", file)
            """
                Perbedaan resize dengan thumbnail adalah 
                resize men-set ukuran citra tepat dengan (image_width, image_height)
                sedangkan thumbnail akan men-set ukuran citra maksimum dengan (image_width, image_height)
            """
            img = img.resize((image_width, image_height), Image.ANTIALIAS)
            # Convert to Numpy Array
            x = img_to_array(img)
            # untuk dapat ditampilkan pada grafik maka harus dinormalkan dengan 255
            x /= 255

            #data = img_to_array(load_img(listAllImages[0]).resize((image_width, image_height), Image.ANTIALIAS))
            #print("Loada file: ", listAllImages[0])
            #print(x.shape)

            dataset[i] = x
            target[i] = self.getCategory(file)
            i += 1
            if i % 150 == 0:
                print("%d images to array" % i)

        print("All images to array!")

        plt.imshow(dataset[100], interpolation='nearest')
        plt.show()

    # membangkitkan dataset secara manual dengan variabel tertentu
    def buildDataset(self, image_width = 130, image_height = 130, channels = 3):
        listAllImages = self.getListOfFiles()

        dataset = np.ndarray(shape=(len(listAllImages), image_width, image_height, channels),
                             dtype=np.float32)
        # array 2 dimensi dengan baris jumlah image dataset, dan kolomnya 2
        target = np.ones(shape=(len(listAllImages), 1), dtype=np.int8)
        listfilename = []

        #print("Build Dataset")
        #print(len(listAllImages))
        #display(_Imgdis(self.getListOfFiles()[1], width=240, height=320))
        #print("Shape dataset: ", dataset.shape)
        #data = np.ndarray((image_width, image_height, channels),dtype=np.float32)

        i=0
        for file in listAllImages:
            #print(os.path.basename(os.path.dirname(dir)))
            img = load_img(file)  # this is a PIL image
            #print("Nama file: ", file)
            """
                Perbedaan resize dengan thumbnail adalah 
                resize men-set ukuran citra tepat dengan (image_width, image_height)
                sedangkan thumbnail akan men-set ukuran citra maksimum dengan (image_width, image_height)
            """
            img = img.resize((image_width, image_height), Image.ANTIALIAS)
            # Convert to Numpy Array
            x = img_to_array(img)
            # untuk dapat ditampilkan pada grafik maka harus dinormalkan dengan 255
            x /= 255

            #data = img_to_array(load_img(listAllImages[0]).resize((image_width, image_height), Image.ANTIALIAS))
            #print("Loada file: ", listAllImages[0])
            #print(x.shape)

            dataset[i] = x
            target[i] = self.getCategory(file)
            listfilename.append(getFileNameFromFilePath(file))
            i += 1
            if i % 150 == 0:
                print("%d images to array" % i)

        print("All images to array!")
        return dataset, target, listfilename

    # Membangkitkan dataset berdasarkan image generator
    def generateDataset(dir='./', input_shape=(150, 150, 3), batch_size=10, counter=1, class_mode='categorical'):

        # image_width, image_height, channels = (130, 130, 3)
        # Jumlah dataset yang dibangkitkan dari direktori citra harus lebih kecil atau sama dengan jumlah citra yang ada
        # pada direktori citra

        train_generator = generateData(train_dir=dir, input_shape=(input_shape[0], input_shape[1]),
                                       batch_size=batch_size,
                                       class_mode=class_mode)
        class_dictionary = train_generator.class_indices

        # print(class_dictionary)

        # countCat = getAllCategoryDir(dir)
        # print("Banyaknya kategory: ", len(countCat))
        # print(countCat)
        dataset = np.ndarray(shape=(counter * batch_size, input_shape[0], input_shape[1], input_shape[2]),
                             dtype=np.float32)
        labels = np.ndarray(shape=(counter * batch_size, len(class_dictionary)), dtype=np.int8)

        i = 0
        start = i
        end = batch_size
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
            end = (i + 1) * batch_size  # karena dimulai dari 0

            if i % counter == 0:
                break
        return (dataset, labels, class_dictionary)

