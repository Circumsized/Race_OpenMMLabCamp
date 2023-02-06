#导入OS模块，该模块提供了非常丰富的函数和方法，用来处理文件和目录
import os

# 定义花朵目标数据集的路径
data_path = '/HOME/scz0auh/run/mmclassification/data/flower_dataset'

#定义标签和每个标签的图像数量
labels = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
num_images = [588, 556, 583, 536, 585]

# 将数据集划分为训练集和验证集，8：2的比例
train_set = []
val_set = []
#循环labels列表，获取每个标签的图像数量
for i in range(len(labels)):
    label_dir = data_path + '/' + labels[i]
    train_set += [label_dir + '/' + f for f in os.listdir(label_dir)][:int(num_images[i] * 0.8)]
    val_set += [label_dir + '/' + f for f in os.listdir(label_dir)][int(num_images[i] * 0.8):]

# 将数据组织成ImageNet格式
train_organize = ''
for f in train_set:
    label = f.split('/')[-2]
    train_organize += f'{f}\t{label}\n'

val_organize = ''
for f in val_set:
    label = f.split('/')[-2]
    val_organize += f'{f}\t{label}\n'

#把训练集中的数据以ImageNet格式写入.txt文件
with open(data_path + '/train.txt', 'w') as output_file:
        output_file.write(train_organize)

with open(data_path + '/val.txt', 'w') as output_file:
        output_file.write(val_organize)