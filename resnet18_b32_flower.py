'''
代码简介
 _base_是一个列表，其中包含三个文件的路径，这三个文件将提供创建模型所必需的基本信息。
 model字典指定要创建的模型的头部参数，包括分类的类别数量和获得的前K个类别。
 data字典指定了使用的数据集的详细信息，其中包括用于训练和测试的数据，以及每个GPU上的样本数和工作者数量。
 optimizer字典指定优化器信息，包括优化器类型、学习率、动量和权重衰减；
 optimizer_config字典指定优化器配置，即梯度裁剪；
 lr_config字典指定学习率策略，即步骤；
 runner字典指定运行器，即使用按照最大epoch数量100进行驱动的EpochBasedRunner，并指定预训练模型的路径。
'''

#定义一个名为_base_的list，list中包含三个路径，分别指向三个.py文件。
_base_ = ['../_base_/models/resnet18.py', '../_base_/datasets/imagenet_bs32.py',
'../_base_/default_runtime.py']

#创建一个名为model的dict
model = dict(
    #head(头部)包含了num_classes参数和topk参数
    head=dict(
        #num_classes参数指定了训练的类别数量
        num_classes=5,
        #topk参数指定了要预测的top k个类别
        topk = (1,)
    ))

#创建了一个名为data的dict，其中包含了4个dict参数
data = dict(
    #指定训练所用样本数量
    samples_per_gpu = 32,
    #指定每个gpu使用worker数量
    workers_per_gpu = 2,
    #指定训练数据集前缀，注解文件和类标文件，测试数据集前缀，注解文件和类标文件
    train = dict(
        data_prefix = 'data/flower/train',
        ann_file = 'data/flower/train.txt',
        classes = 'data/flower/classes.txt'
    ),
    val = dict(
    data_prefix = 'data/flower/val',
    ann_file = 'data/flower/val.txt',
    classes = 'data/flower/classes.txt'
    )
)

'''
创建了一个名为optimizer的dict，其中包含type, lr, momentum, weight_decay四个参数，
type参数指定了优化器的类型，lr参数指定了学习率，
momentum参数指定了动量参数，weight_decay参数指定了L2正则化系数
'''
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
'''
创建了一个名为optimizer_config字典，
其中包含grad_clip参数，grad_clip参数用于控制梯度更新时的最大范数
'''
optimizer_config = dict(grad_clip=None)

'''
创建了一个名为lr_config的dict，其中包含policy和step两个参数，
policy参数指定了学习率变化规则，step参数表示学习率变化的步数
'''
lr_config = dict(
    policy='step',
    step=[1])

'''
创建了一个名为runner的dict，其中包含type和max_epochs两个参数，
type参数指定了运行器的类型，max_epochs参数指定了训练最大的epoch次数
'''
runner = dict(type='EpochBasedRunner', max_epochs=100)


 # 预训练模型
 #定义了load_from变量，该变量指定了预训练的模型文件的路径。
load_from ='/HOME/scz0auh/run/mmclassification/checkpoints/resnet18_batch256_imagenet_20200708-34ab8f90.pth'
