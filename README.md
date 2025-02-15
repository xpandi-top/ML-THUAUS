# ML-THUAUS 
## 问题分析 ![](https://img.shields.io/badge/status-empty-red.svg)
1. 分类问题
2. 回归问题
3. 评价函数，accuracy， recall， AUC 


## 数据概览-可视化 overall_data.py ![](https://img.shields.io/badge/status-compelete-green.svg)
1. 数据的类型： 类别， 日期， 数值，ID
2. 数据的分布：每个特征下的数据分布
3. 数据之间的关系：correlation

## 数据预处理- preprocessing.py ![](https://img.shields.io/badge/status-compelete-green.svg)
1. 空值处理， 空值填充， 与删除 impute missing values
[SkLearn](https://scikit-learn.org/stable/modules/impute.html#impute)

2. 根据不同的数据类型，将数据进行转化， one hot ，ordinary

3. 数据归一化：standard scale， minmax scale
[SKlearn](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing)

4. 用pipeline串联

## 特征工程- preprocessing.py ![](https://img.shields.io/badge/status-incompelete-yellow.svg)
1. 降维， PCA， LDA, sklearn feature_selection
2. 特征生成，比如log， ➗， ➕，✖️

## 拆分测试集与训练集- preprocessing.py

sklearn split 

### 样本不均匀情况处理 ![](https://img.shields.io/badge/status-compelete-green.svg)
1. Under sample
2. Up sample
3. Smote sample


## 模型选择 *_model.py( 包括grid search) ![](https://img.shields.io/badge/status-incompelete-yellow.svg)
- [x] Xgboost
- [x] Random Forest
- [ ] Ada
- [ ] Et
- [ ] Svc
- [ ] rf
- [ ] Light gbm

### 针对不均匀样本的：unbalanced_model.py ![](https://img.shields.io/badge/status-empty-red.svg)
1. Svm-one class
sklearn
2. Cost sensitive algorithm
costla

### 参数选择 ![](https://img.shields.io/badge/status-compelete-green.svg)
Grid search

## 模型融合- model_ensembling.py ![](https://img.shields.io/badge/status-incompelete-yellow.svg)
Ensembling

## 结果保存 - final_visualization.py ![](https://img.shields.io/badge/status-empty-red.svg)
1. 模型结构呈现
2. 模型参数保存
3. 数据结果保存
4. 各特征贡献概览-可视化
