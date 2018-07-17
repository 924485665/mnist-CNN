# 这个ipython notebook主要是我解决Kaggle Titanic问题的思路和过程

import pandas as pd  # 数据分析
import numpy as np  # 科学计算
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.learning_curve import learning_curve
from sklearn.ensemble import BaggingRegressor

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号




data_train = pd.read_csv("D:/titanic/train.csv")
# print(data_train.columns)
# print(data_train.info())
# print(data_train.describe())

### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age','Fare','Parch','SibSp','Pclass']]
    print(age_df)

    # 乘客分成已知年龄和未知年龄两部分
    known_age =  age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    # print(unknown_age)
    # print(unknown_age[:,1:])
    # print(unknown_age[:,1::])


    # y即目标年龄

    y = known_age[:,0]
    # print(y)

    # X即特征属性值
    X = known_age[:,1:]
    # print(X)

    rfr = RandomForestRegressor(random_state = 0, n_estimators = 2000,n_jobs = -1)
    rfr.fit(X,y)
    print(rfr)
    predictAges = rfr.predict(unknown_age[:,1:])  #缺失的Age值的预测结果，用列表表示

    df.loc[(df.Age.isnull()),'Age'] = predictAges  # 用得到的预测结果填补原缺失数据
    # print(df.loc[:,'Age'])

    return df, rfr


def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()),'Cabin'] = 'Yes'
    df.loc[(df.Cabin.isnull()),'Cabin'] = 'No'
    # print(df.loc[:,'Cabin'])
    return df



# 用sklearn的learning_curve得到training_score和cv_score，使用matplotlib画出learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"训练样本数")
        plt.ylabel(u"得分")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"交叉验证集上得分")

        plt.legend(loc="best")

        plt.draw()
        plt.show()
        plt.gca().invert_yaxis()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff



if __name__ == '__main__':
    data_train, rfr = set_missing_ages(data_train)
    data_train = set_Cabin_type(data_train)
    # print(data_train)
    #
    # # 因为逻辑回归建模时，需要输入的特征都是数值型特征
    # # 我们先对类目型的特征离散/因子化 ONE-HOT
    # # 以Cabin为例，原本一个属性维度，因为其取值可以是['yes','no']，而将其平展开为'Cabin_yes','Cabin_no'两个属性
    # # 原本Cabin取值为yes的，在此处的'Cabin_yes'下取值为1，在'Cabin_no'下取值为0
    # # 原本Cabin取值为no的，在此处的'Cabin_yes'下取值为0，在'Cabin_no'下取值为1
    # # 我们使用pandas的get_dummies来完成这个工作，并拼接在原来的data_train之上，如下所示
    # dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
    # dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
    # dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
    # dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')
    #
    # df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    # df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    #
    # # 接下来我们要接着做一些数据预处理的工作，比如scaling，将一些变化幅度较大的特征化到[-1,1]之内
    # # 这样可以加速logistic regression的收敛
    # scaler = preprocessing.StandardScaler()
    # age_scale_param = scaler.fit(df['Age'])
    # df['Age_scaled'] = scaler.fit_transform(df['Age'], age_scale_param)
    # fare_scale_param = scaler.fit(df['Fare'])
    # df['Fare_scaled'] = scaler.fit_transform(df['Fare'], fare_scale_param)
    #
    # # 我们把需要的feature字段取出来，转成numpy格式，使用scikit-learn中的LogisticRegression建模
    # train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    # train_np = train_df.as_matrix()
    # print('最终训练集：')
    # print(train_df)
    # print(train_np)
    #
    # # y即Survival结果
    # y = train_np[:, 0]
    #
    # # X即特征属性值
    # X = train_np[:, 1:]
    # print(X)
    # print(X.shape)
    # print(type(X))
    #
    # # fit到LogisticRegression之中
    # clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    # clf.fit(X, y)
    # print(clf)
    #
    # # 接下来咱们对训练集和测试集做一样的操作
    # data_test = pd.read_csv('D:/titanic/test.csv')
    # data_test.loc[(data_test.Fare.isnull()), 'Fare'] = 0
    # # 接着我们对test_data做和train_data中一致的特征变换
    # # 首先用同样的RandomForestRegressor模型填上丢失的年龄
    # tmp_df = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    # null_age = tmp_df[data_test.Age.isnull()].as_matrix()
    # # 根据特征属性X预测年龄并补上
    # X = null_age[:, 1:]
    # predictedAges = rfr.predict(X)
    # data_test.loc[(data_test.Age.isnull()), 'Age'] = predictedAges
    #
    # data_test = set_Cabin_type(data_test)
    # dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')
    # dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix='Embarked')
    # dummies_Sex = pd.get_dummies(data_test['Sex'], prefix='Sex')
    # dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix='Pclass')
    #
    # df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    # df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    # df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'], age_scale_param)
    # df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'], fare_scale_param)
    #
    # # 做预测取结果
    # test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    # print('最终测试集')
    #
    # predictions = clf.predict(test.as_matrix())
    # print('测试结果为:')
    # print(predictions)
    # result = pd.DataFrame(
    #     {'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
    # result.to_csv("D:/titanic/logistic_regression_predictions.csv", index=False)
    #
    # # # 简单看看打分情况
    # # clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    # # all_data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    # # X = all_data.as_matrix()[:, 1:]
    # # y = all_data.as_matrix()[:, 0]
    # # print(cross_validation.cross_val_score(clf, X, y, cv=5))
    #
    # # plot_learning_curve(clf, u"学习曲线", X, y)
    #
    # train_df = df.filter(
    #     regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
    # print('the len of traindf is {}'.format(len(train_df.columns.values)))
    # train_np = train_df.as_matrix()
    #
    # # y即Survival结果
    # y = train_np[:, 0]
    #
    # # X即特征属性值
    # X = train_np[:, 1:]
    #
    # # # fit到BaggingRegressor之中
    # # clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    # # bagging_clf = BaggingRegressor(clf, n_estimators=10, max_samples=0.8, max_features=1.0, bootstrap=True,
    # #                                bootstrap_features=False, n_jobs=-1)
    # # bagging_clf.fit(X, y)
    # # print(cross_validation.cross_val_score(clf, X, y, cv=10))
    # #
    # # test = df_test.filter(
    # #     regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
    # # predictions = bagging_clf.predict(test)
    # # result = pd.DataFrame(
    # #     {'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
    # # result.to_csv("D:/titanic/logistic_regression_predictions1.csv", index=False)
    # # np.where