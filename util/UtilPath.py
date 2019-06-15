import os, shutil
import tkinter as tk
from tkinter import filedialog

def getDirectoryPath():
    root = tk.Tk()
    root.withdraw()
    data_dir = filedialog.askdirectory(parent=root, initialdir="//", title='Pick a directory')
    return data_dir

def getFileName():
    root = tk.Tk()
    root.withdraw()
    root.filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                               filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    return root.filename

def getAllFiles(dir):
    return os.listdir(dir)


def getAllCategoryDir(dir):
    listOfCat = list()
    listOfCat = [x for x in os.listdir(dir) if not x.__contains__('.')]

    return listOfCat

# mengambalikan semua file yang ada pada direktori tertentu sesuai dengan extensi yang akan diambil lengkap dengan path file tersebut
def getListOfFileDir(dir='.', ext='.'):
    listOfFiles = list()
    listOfFiles = [os.path.join(dp, f) for dp, dn, filenames in os.walk(dir) for f in filenames if
                   os.path.splitext(f)[1] == ext]
    """
    for root, dirs, files in os.walk(dir, topdown=False):
        listOfFiles += [os.path.join(root, file) for file in files]
    """

    return listOfFiles

# mengembalikan nama direktori terkahir, misalkan direktory yang diberikan /doc/dir1/dir2/image.jpg
# makan akan mengembalikan dir2
def getLastDirectoryPath(dir):
    return os.path.basename(os.path.dirname(dir))

def getFileNameFromFilePath(inputFilepath):
    return os.path.basename(inputFilepath)


def create_dir_dataset(base_dir, ds_name="ds"):
    base_dir = os.path.join(base_dir, ds_name)
    os.mkdir(base_dir)
    train_dir = os.path.join(base_dir, 'train')
    os.mkdir(train_dir)
    validation_dir = os.path.join(base_dir, 'validation')
    os.mkdir(validation_dir)
    test_dir = os.path.join(base_dir, 'test')
    os.mkdir(test_dir)


# base_dir adalah direktori dataset, yaitu direktori sebelum catageori dataset
# misalkan untuk training, terdapat kategori Cherry, maka base_dir nya adalah training
def create_dir_category_ds(base_dir_from, base_dir_to, list_category):
    print("list direktori")
    for cat_name in list_category:
        print(cat_name)
        base_dir = os.path.join(base_dir_to, cat_name)
        os.mkdir(base_dir)


# base_dir adalah direktori dataset, yaitu direktori sebelum catageori dataset
# misalkan untuk training, terdapat kategori Cherry, maka base_dir nya adalah training
def create_dir_category_ds_target(base_dir_from, base_dir_to):
    print("list direktori")
    list_category = getAllCategoryDir(base_dir_from)
    for cat_name in list_category:
        print(cat_name)
        base_dir = os.path.join(base_dir_to, cat_name)
        os.mkdir(base_dir)


def copy_file_from_dir_to_dir(base_dir_from, base_dir_to):
    print("Copy file")
    # 1. get list file name
    listfilename = getAllFiles(base_dir_from)
    listfilename = [x for x in listfilename if not x.startswith('.')]
    for filename in listfilename:
        src = os.path.join(base_dir_from, filename)
        dst = os.path.join(base_dir_to, filename)
        shutil.copyfile(src, dst)


"""
    Tujuannya untuk meng-copy semua file yang ada pada direktori tertentu ke direktori tujuan
    Misalkan: ingin meng-copy semua file *.jpg yang ada pada direktori train.
    Pada direktori train mengandung kategori apple, manggo, banana
    Pada direktori apple juga mengandung 100 citra *.jpg
    Maka tujuan method ini adalah untuk meng-copy semua file yang ada di train dataset ke direktori train target
    Langkah-langkahnya:
    1. Pilih direktori awal (dir_from)
    2. Pilih direktori tujuan (dir_target)
    3. Ambil semua direktori yang terkandung pada direktori dir_from, dengan menggunakan @getAllCategoryDir(base_dir_from)
    4. Buat direktori baru berdasarkan kategorinya yang diambil dari tahap 3
    Proses 3 dan 4 ini dapat dilakukan secara langsung menggunakan @create_dir_category_ds_

"""
def copy_dataset(base_dir_from, base_dir_to):
    # 1. mengambil semua kategori
    list_category = getAllCategoryDir(base_dir_from)
    list_dir_dst_category = list()
    list_dir_scr_catgeory = list()
    # 2. membuat direktori baru pada direktori target berdasarkan nama dataset base_dir_to
    for cat_name in list_category:
        base_dir = os.path.join(base_dir_to, cat_name)
        os.mkdir(base_dir)
        # 3. menyimpan list nama direktory target baru yang dibentuk
        list_dir_dst_category.append(base_dir)
        list_dir_scr_catgeory.append(os.path.join(base_dir_from, cat_name))

    for (dir_cat_from, dir_cat_to) in zip(list_dir_scr_catgeory, list_dir_dst_category):
        listfilename = getAllFiles(dir_cat_from)
        listfilename = [x for x in listfilename if not x.startswith('.')]

        for filename in listfilename:
            src = os.path.join(dir_cat_from, filename)
            dst = os.path.join(dir_cat_to, filename)
            shutil.copyfile(src, dst)


def copy_file_dataset(base_dir_from, base_dir_to, start=0, end=1):
    fnames = ['{}.jpg'.format(i) for i in range(start, end)]
    for fname in fnames:
        src = os.path.join(base_dir_from, fname)
        dst = os.path.join(base_dir_to, fname)
        shutil.copyfile(src, dst)

def main():
    dirpath = os.getcwd()
    print("Path name = " + dirpath)
    original_dataset_dir = '/Users/jasmanpardede/Documents/Jasman/S3/DataSet/kagglecatsanddogs_3367a/PetImages/'
    

if __name__ == "__main__":
    main()