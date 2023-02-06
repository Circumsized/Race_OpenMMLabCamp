#设置预训练图像分类模型
model = dict(
    type='ImageClassifier',
    # 将要使用的backbone模型是：ResNet，深度为18，分解为4步，输出的索引是3，样式是pytorch
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    #将要使用的neck模型是：GlobalAveragePooling
    neck=dict(type='GlobalAveragePooling'),

    #头部模型是LinearClsHead
    head=dict(
        type='LinearClsHead',
        #有五个类别
        num_classes=5,
        #输入通道为512
        in_channels=512,
        #损失函数类型为CrossEntropyLoss，权重为1.0
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        #输出前1个结果
        topk=(1, )))

#输出数据集类型是ImageNet
dataset_type = 'ImageNet'
#图像规范化配置，均值[123.675, 116.28, 103.53]，标准差[58.395, 57.12, 57.375]，转换成RGB图像
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


#训练管道，从文件中读取图像
train_pipeline = [
    dict(type='LoadImageFromFile'),
    #随机裁剪大小为224
    dict(type='RandomResizedCrop', size=224),
    #垂直翻转概率为0.5
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    #规范化均值[123.675, 116.28, 103.53]，标准差[58.395, 57.12, 57.375]，
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    #将图片转换成tensor，将标签也转换成tensor，并将图像和标签收集起来
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

'''
测试管道，从文件中读取图片
调整大小为256，从中心裁剪大小为224
标准化后转换成tensor，收集起来
'''
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    #每个GPU上每次迭代处理32个样本
    samples_per_gpu=32,
    #每个GPU上使用2个子进程加载数据
    workers_per_gpu=2,
    train=dict(
        #指定训练数据集类型
        type='ImageNet',
        #指定训练数据存放位置
        data_prefix='data/flower/train',

        pipeline=[
            #从文件中加载图像
            dict(type='LoadImageFromFile'),
            #随机裁剪大小为224
            dict(type='RandomResizedCrop', size=224),
            #水平方向翻转概率为0.5
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            #标准化图像均值[123.675, 116.28, 103.53]，标准差[58.395, 57.12, 57.375]，转换成RGB格式
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),

            '''
            将图像和标签标准化后转换成tensor，收集起来
            '''
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ],
        #指定训练集的数据文件
        ann_file='data/flower/train.txt',
        #指定的预测分类
        classes='data/flower/classes.txt'),
    #定义评估参数，包括数据源，图像处理管道等
    val=dict(
        #指定数据类型
        type='ImageNet',
        #指定验证集数据文件路径
        data_prefix='data/flower/val',
        #指定验证集数据文件
        ann_file='data/flower/val.txt',
        #定义图像处理管道，包括读取图片、缩放、裁剪等
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(256, -1)),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                #数值为RGB三个通道的均值
                mean=[123.675, 116.28, 103.53],
                #数值为RGB三个通道的标准差
                std=[58.395, 57.12, 57.375],
                #将图像转换为RGB格式
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ],

        classes='data/flower/classes.txt'),
    test=dict(
        type='ImageNet',
        data_prefix='data/imagenet/val',
        ann_file='data/imagenet/meta/val.txt',
        pipeline=[
            #从文件加载图像
            dict(type='LoadImageFromFile'),
            #将图像大小调整为 256 x -1 (高度保持不变)
            dict(type='Resize', size=(256, -1)),
            #从图像中心裁剪 224 x 224 像素的子图像
            dict(type='CenterCrop', crop_size=224),
            dict(
                #对图像像素进行normalization，使用给定的mean和std，并将BGR顺序转换为RGB顺序。
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]))

#设置评估间隔和度量标准
evaluation = dict(interval=1, metric='accuracy')
#设置间隔点的保存间隔
checkpoint_config = dict(interval=1)
#设置日志记录间隔并加载 TextLoggerHook 类型的hooks
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
#设置使用 NCCL 作为后端来运行分布式程序
dist_params = dict(backend='nccl')
#设置日志级别
log_level = 'INFO'
#从指定路径加载检查点
load_from = '/HOME/scz0auh/run/mmclassification/checkpoints/resnet18_batch256_imagenet_20200708-34ab8f90.pth'
#设置是否恢复训练
resume_from = None
#设置作业为训练，并设置作业分组为1组
workflow = [('train', 1)]

#使用 SGD 优化器，设置学习率为 0.001，动量为 0.9，权重衰减系数为 0.0001
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
#设置梯度裁剪操作
optimizer_config = dict(grad_clip=None)
#设置学习率调整策略，并在第一个epoch调整学习率
lr_config = dict(policy='step', step=[1])
#设置运行器类型为 EpochBasedRunner，并设置最大epoch数为 100
runner = dict(type='EpochBasedRunner', max_epochs=100)
#指定工作目录
work_dir = 'work/resnet18_b32_flower'
#使用0号GPU进行训练
gpu_ids = [0]
