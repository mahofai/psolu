# AutoML4Bio

## intro:
autoMM.py脚本调用autogluon 框架下HPO及相关功能实现multimodel predictor的训练及基本hpo
主要脚本：autoMM.py, MLproject,必须在项目的automl conda环境下使用

！数据集仅支持包含有序列或数据的csv文件，其他形式的数据集暂不支持


### 参数说明:
基本输入

    --target_column：预测目标列

    --metric： 评估指标 
    二分类: ["acc","accuracy","log_loss","roc_auc"]

    多分类: ["acc","accuracy","log_loss"]

    回归: ["root_mean_squared_error","r2","spearmanr","pearsonr"]

    --train_data：训练数据集csv文件名 

    --valid_data: 测试数据集文件名

    --checkpoint_name: 输入hugging face模型目录名

    --tabular:  0:机器学习 1:深度学习 

超参搜索 设定
    --mode: 预设参数设置(medium_quality/ best_quality/ manual), 
    除手动模式外均使用预设参数搜索范围，best/medium quality仅在num_trials上有差异

    --searcher: 超参搜索算法（bayes/grid/random）

    --num_trials: 贝叶斯优化和随机优化尝试的次数，网格优化默认遍历所有数据点组合不被此参数影响

    --max_epochs: 模型每次调优的最大epochs

参数搜索范围 (因mlflow run -P 传参只支持single value，现阶段在平台输入时必须带上 “” 以将数据以str 形式输入，autoMM.py 内会拆分处理）

    * 贝叶斯优化时输入的为“min max”，数字较小的必须在前，网格搜索时可输入多个数据，用空格间隔

    --lr : 学习率

    --lr_decay: 学习率衰减

    --weight_decay: 权重衰减

    --batch_size：批次样本数量，离散值，可输入多个数据，用空格间隔

    --optim_type:  优化器 (adam/ adamw/ sgd) 

    --lr_schedule: 学习率衰减方法(cosine_decay/ 
    polynomial_decay/ linear_decay)

### 数据特征集文件格式：

所有数据必须以data.zip 形式上传，所有模型必须以model.zip格式上传

data.zip

    soluprot_train.csv

    soluprot_valid.csv

    soluprot_test.csv


model.zip

    esm2_8m

### 测试案例:

automl机器学习 测试：mlflow run psolu -P train_data=soluprot_train.csv -P valid_data=soluprot_valid.csv -P target_column=solubility -P tabular=1 -P metric=roc_auc -P mode=manual

深度学习测试：mlflow run psolu -P check_point_name=esm2_8m -P train_data=soluprot_train.csv -P valid_data=soluprot_valid.csv -P target_column=solubility -P lr=0.001 -P lr_decay=0.002 -P weight_decay=0.003 -P batch_size=32 -P optim_type=adam -P lr_schedule=cosine_decay -P mode=manual -P metric=roc_auc -P max_epochs=5

小规模贝叶斯超参数搜索：mlflow run psolu -P check_point_name=esm2_8m -P train_data=soluprot_train.csv -P valid_data=soluprot_valid.csv -P target_column=solubility -P mode=medium_quality -P metric=roc_auc -P max_epochs=2

大规模贝叶斯超参数搜索（耗时为小规模的20倍）：mlflow run psolu -P check_point_name=esm2_8m -P train_data=soluprot_train.csv -P valid_data=soluprot_valid.csv -P target_column=solubility -P mode=best_quality -P metric=roc_auc -P max_epochs=2

手动贝叶斯超参数搜索：mlflow run psolu -P check_point_name=esm2_8m -P train_data=soluprot_train.csv -P valid_data=soluprot_valid.csv -P target_column=solubility -P lr='0.001,0.1' -P lr_decay='0.002,0.2' -P weight_decay='0.003,0.3' -P batch_size=32 -P optim_type=adam -P lr_schedule=cosine_decay -P mode=manual -P metric=roc_auc -P max_epochs=2 -P num_trials=2 -P searcher=bayes

手动随机超参数搜索：mlflow run psolu -P check_point_name=esm2_8m -P train_data=soluprot_train.csv -P valid_data=soluprot_valid.csv -P target_column=solubility -P lr='0.001,0.1' -P lr_decay='0.002,0.2' -P weight_decay='0.003,0.3' -P batch_size=32 -P optim_type=adam -P lr_schedule=cosine_decay -P mode=manual -P metric=roc_auc -P max_epochs=2 -P num_trials=2 -P searcher=random

手动随机超参数搜索：mlflow run psolu -P check_point_name=esm2_8m -P train_data=soluprot_train.csv -P valid_data=soluprot_valid.csv -P target_column=solubility -P lr='0.001,0.1' -P lr_decay='0.002,0.2' -P weight_decay='0.003,0.3' -P batch_size=32 -P optim_type=adam -P lr_schedule=cosine_decay -P mode=manual -P metric=roc_auc -P max_epochs=2  -P searcher=grid


### 注意事项:

把（处理序列的hugging face）模型和数据集打包为一个zip，上传到数据特征集，当跑算法时选择数据特征集，特征集会解压到当前目录的上一级并保留.zip作最外层目录

目前超参设置/优化暂不支持tabular predictor（机器学习），但当mode 设置为 “best quality” 时tabular predictor会使用集成学习（stacking + bagging）的方法尝试提高表现



### TODO

auto feature engineering

HPO for single/multiple tabular model

tabular predictor select model

autoMM HPO精度测试
