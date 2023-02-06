import os
import random
from collections import OrderedDict

def copy(src, target):
    with open(src, 'rb') as f:
        data = f.read()
    with open(target, 'wb') as f:
        f.write(data)

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)



if __name__ == '__main__':
    print(os.listdir('.\\flower_dataset\\'))

    train_rate = 0.8

    raw_dir = '.\\flower_dataset\\'

    root_dir = 'flower_dataset_Image_Net_4_1'
    train_dir = os.path.join(root_dir, 'train')
    val_dir = os.path.join(root_dir, 'val')
    train_txt = os.path.join(root_dir, 'train.txt')
    val_txt = os.path.join(root_dir, 'val.txt')
    class_txt = os.path.join(root_dir, 'class.txt')

    mkdir(root_dir)
    mkdir(train_dir)
    mkdir(val_dir)


    classes = os.listdir(raw_dir)

    train_data = OrderedDict()
    val_data = OrderedDict()

    for id in range(len(classes)):
        cls = classes[id]

        train_cls_dir = os.path.join(train_dir, cls)
        val_cls_dir = os.path.join(val_dir, cls)

        mkdir(train_cls_dir)
        mkdir(val_cls_dir)

        train_data_cls = []
        val_data_cls = []

        class_dir = os.path.join(raw_dir, cls)
        images = os.listdir(class_dir)
        images.sort()

        random.shuffle(images)

        train_num = int(len(images) * train_rate)
        for i in range(train_num):
            copy(os.path.join(class_dir, images[i]), os.path.join(train_cls_dir, images[i]))
            train_data_cls.append(os.path.join(cls, images[i]))
        for i in range(train_num, len(images)):
            copy(os.path.join(class_dir, images[i]), os.path.join(val_cls_dir, images[i]))
            val_data_cls.append(os.path.join(cls, images[i]))
        
        train_data[cls] = train_data_cls
        val_data[cls] = val_data_cls


    with open(train_txt, 'w') as f:
        for i in range(len(classes)):
            for image in train_data[classes[i]]:
                f.write(f"{image} {i}\n")

    with open(val_txt, 'w') as f:
        for i in range(len(classes)):
            for image in val_data[classes[i]]:
                f.write(f"{image} {i}\n")

    with open(class_txt, 'w') as f:
        for i in range(len(classes)):
            f.write(classes[i] + '\n')
        

