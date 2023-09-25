# AutoML4Bio

## soluprot intro:
autoMM.py脚本调用autogluon 框架下HPO及相关功能实现multimodel predictor的训练及基本hpo
主要脚本：autoMM.py, MLproject
必须在项目的automl conda环境下使用

### 参数说明:
基本输入

    --target_column：预测目标列

    --metric： 评估指标

    --data：数据集路径

    --tabular:  0:机器学习 1:深度学习 

HPO 设定
    --mode: 预设参数设置(medium_quality/ best_quality/ manual), 
    除手动模式外均使用预设参数搜索范围，best/medium quality仅在num_trials上有差异

    --test_n_fold:  选择手动划分的第n折作验证集,（column name: “fold”)，为 -1 时做随机划分，训练集80%，验证集20% 

    --searcher: 优化算法（bayes/grid/random）

    --num_trials: 贝叶斯优化和随机优化尝试的次数，网格优化默认遍历所有数据点组合不被此参数影响

    --check_point_name: 输入hugging face模型路径

    --max_epochs: 模型每次调优的最大epochs

参数搜索范围 (因mlflow run -P 传参只支持single value，现阶段在平台输入时必须带上 “” 以将数据以str 形式输入，autoMM.py 内会拆分处理）
    --lr : 贝叶斯优化时输入的为“min max”，数字较小的必须在前，网格搜索时可输入多个数据，用空格间隔

    --lr_decay: 同上

    --weight_decay: 同上

    --batch_size：离散值，可输入多个数据，用空格间隔

    --optim_type:  优化器 (adam/ adamw/ sgd) 

    --lr_schedule: 学习率衰减方法(cosine_decay/ 
    polynomial_decay/ linear_decay)


### 测试案例:
无hpo： python autoMM.py --mode manual


预设短时间hpo： python autoMM.py  --mode medium_quality


预设长时间hpo:    python autoMM.py --mode best_quality


bayes 手动hpo：python autoMM.py --mode manual --searcher bayes --lr "0.0001,0.1" --lr_decay "0.0002,0.2"  --weight_decay "0.0003,0.3" --batch_size "16,32,64" --lr_schedule "cosine_decay" --optim_type "adam"


grid search 手动hpo：python autoMM.py --mode manual --searcher grid --lr "0.0001,0.01,0.1" --lr_decay "0.0002,0.02,0.2"  --weight_decay "0.0003,0.03,0.3" --batch_size "16,32,64" --lr_schedule "cosine_decay" --optim_type "adam"


机器学习集成:  python autoMM.py  --mode medium_quality --tabular 1





### 注意事项:
csv数据集应当有一个"split" columnn把数据分为 train/test/valid

把（处理序列的hugging face）模型和数据集打包为一个zip，上传到数据特征集，当跑算法时选择数据特征集，特征集会解压到当前目录的上一级

目前超参设置/优化暂不支持tabular predictor（机器学习），但当mode 设置为 “best quality” 时tabular predictor会使用集成学习（stacking + bagging）的方法尝试提高表现




### TODO

auto feature engineering

HPO for single/multiple tabular model

tabular predictor select model

autoMM HPO精度测试
