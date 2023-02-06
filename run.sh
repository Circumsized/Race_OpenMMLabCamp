'''
#!/bin/bash 是一个标准的shell脚本，用于指示解释器执行该脚本。
'''
#!/bin/bash

# 加载模块
module load anaconda/2021.05
module load cuda/11.1
module load gcc/7.3

# 激活环境(name==mmclassification)
source activate mmclassification

# 刷新日志缓存
export PYTHONUNBUFFERED=1

# 训练模型
'''
执行python脚本tools/train.py
传入configs/resnet18/resnet18_b32_flower.py参数
在work/resnet18_b32_flower目录下训练模型
'''

python tools/train.py \
       configs/resnet18/resnet18_b32_flower.py \
       --work-dir work/resnet18_b32_flower