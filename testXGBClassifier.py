# -*- coding: utf-8 -*-
### load module
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from xgboost import plot_importance
import pickle

def object2Category(sample):
    # failed
    optimized_sam = sample.copy()
    optimized_sam = optimized_sam.convert_objects(convert_dates=True,convert_numeric=True,convert_timedeltas=True,copy=True)
    return optimized_sam 

def object2OneHot(sample):
    # failed
    # Using the get_dummies will create a new column for every unique string in a certain column
    # print(sample.columns)
    return pd.get_dummies(sample) 

def label2Integer(sample):
    num = sample.shape[1]
    sample = sample.fillna(0).copy()
    for i in range(num):
        dimensional_name = sample.columns[i]
        sample[dimensional_name] = LabelEncoder().fit_transform(sample[dimensional_name].astype(str))
    return sample

def plotROC(y_test,y_pred):
    # Compute ROC curve and ROC area for each class
    n_classes = 2    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test, y_score)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == '__main__':
    ### load datasets
    # digits = datasets.load_digits()
    sample = pd.read_csv('sub_sample.csv')

    ### data analysis
    # print(sample.head())

    ## data split
    # x_train,x_test,y_train,y_test = train_test_split(digits.data,
    #                                                  digits.target,
    #                                                  test_size = 0.3,
    #                                                  random_state = 33)

    # print(type(sample.drop(['Tag'],axis=1)))
    # print(sample.drop(['Tag'],axis=1).head())
    ## optimize sample dytype object to category
    optimized_sam = label2Integer(sample)
    # print(optimized_sam.head())
    # print(type(optimized_sam))
    # print(type(optimized_sam.drop(['Tag'],axis=1)))
    # print(optimized_sam.drop(['Tag'],axis=1).head())



    x_train,x_test,y_train,y_test = train_test_split(
                                                    optimized_sam.drop(['Tag'],axis=1),
                                                    optimized_sam['Tag'], 
                                                    test_size = 0.3,
                                                    random_state = 33)
    ### fit model for train data
    model = XGBClassifier(learning_rate=0.3,
                          n_estimators=10,         # 树的个数--1000棵树建立xgboost
                          max_depth=6,               # 树的深度
                          min_child_weight = 1,      # 叶子节点最小权重
                          gamma=0.,                  # 惩罚项中叶子结点个数前的参数
                          subsample=0.8,             # 随机选择80%样本建立决策树
                          colsample_btree=0.8,       # 随机选择80%特征建立决策树
                          objective='multi:softmax', # 指定损失函数
                          num_class=2,#指定分类数量
                          scale_pos_weight=1,        # 解决样本个数不平衡的问题
                          random_state=27            # 随机数
                          )
    model.fit(x_train,
              y_train,
              eval_set = [(x_test,y_test)],
              eval_metric = "mlogloss",
              early_stopping_rounds = 10,
              verbose = True)

    ### save the model
    file = open("model.pickle", "wb")
    pickle.dump(model, file)
    file.close()

    ### plot feature importance
    fig,ax = plt.subplots(figsize=(15,15))
    plot_importance(model,
                    height=0.5,
                    ax=ax,
                    max_num_features=64)
    plt.show()

    ### make prediction for test data
    y_pred = model.predict(x_test)
    # print(type(y_pred))

    ### model evaluate
    accuracy = accuracy_score(y_test,y_pred)
    # print(list(y_test))
    # print(list(y_pred))
    print("accuarcy: %.2f%%" % (accuracy*100.0))
    # plotROC(y_test,y_pred)