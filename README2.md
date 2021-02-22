# 1.标注数据生成

1、执行data.py，在data/origin生成test、train、valid标注数据集;

# 2.训练

1、执行main.py，训练结束后生成checkpoints/xxx.pth权重文件;

# .3.预测

1、修改predict.py中的fp变量，指向训练得到的权重文件；  
2、执行predict.py，根据console提示进行预测；