#coding:utf-8
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
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
from sklearn.learning_curve import learning_curve
from sklearn import cross_validation
from sklearn.learning_curve import learning_curve
from sklearn.ensemble import BaggingRegressor
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
import time
import os
from sklearn.feature_extraction.text import TfidfTransformer
import warnings
warnings.filterwarnings('ignore')


# x = np.arange(100)
# fig = plt.figure()
# fig.set(alpha = 0.2)
#
# ax1 = fig.add_subplot(1,4,1)
# ax1.plot(x,x)
#
# ax2 = fig.add_subplot(1,4,2)
# ax2.plot(x, -x)
#
# ax3 = fig.add_subplot(1,4,3)
# ax3.plot(x, x ** 2)
#
# ax4 = fig.add_subplot(1,4,4)
# ax4.plot(x, np.log(x))
#
# plt.show()

#各种舱级别情况下各性别的获救情况
data_train = pd.read_csv("D:/titanic/train.csv")
# fig = plt.figure()
# fig.set(alpha = 0.65)
# plt.title(u'根据舱位等级和性别来看获救情况')
#
# ax1= fig.add_subplot(141)
# data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind = 'bar',label = 'female high_class',color = 'red')
# ax1.set_xticklabels([u'获救',u'未获救'],rotation = 30)
# ax1.legend([u'女性/高级舱'],loc = 'best')
#
# ax2=fig.add_subplot(142, sharey=ax1)
# data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')
# ax2.set_xticklabels([u"未获救", u"获救"], rotation=0)
# plt.legend([u"女性/低级舱"], loc='best')
#
# ax3=fig.add_subplot(143, sharey=ax1)
# data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')
# ax3.set_xticklabels([u"未获救", u"获救"], rotation=0)
# plt.legend([u"男性/高级舱"], loc='best')
#
# ax4=fig.add_subplot(144, sharey=ax1)
# data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')
# ax4.set_xticklabels([u"未获救", u"获救"], rotation=0)
# plt.legend([u"男性/低级舱"], loc='best')
#
# plt.show()



# g = data_train.groupby(['SibSp','Survived'])
# print(pd.DataFrame(g.count()['PassengerId']))

# df = pd.DataFrame({'key1':list('aabba'),
#                   'key2': ['one','two','one','two','one'],
#                   'data1': np.random.randn(5),
#                   'data2': np.random.randn(5)})
# print(df.groupby(['key1']).count())

# print(data_train.Cabin.value_counts())

# survived_1 = data_train[data_train.Cabin.notnull()][data_train.Survived == 1].value_counts()
# survived_0 = data_train[data_train.Cabin.notnull()][data_train.Survived == 0].value_counts()
# df = pd.DataFrame({u'未获救':survived_0,u'获救':survived_1})
# df.plot(kind = 'bar',stacked = True)
# plt.title(u'按Cabin有无看获救情况')
# plt.ylabel(u'人数')
# plt.xlabel(u'Cabin有无')
# plt.show()


#cabin的值计数太分散了，绝大多数Cabin值只出现一次。感觉上作为类目，加入特征未必会有效
#那我们一起看看这个值的有无，对于survival的分布状况，影响如何吧
# survived_cabin = data_train.Survived[data_train.Cabin.notnull()].value_counts()
# survived_no_cabin = data_train.Survived[data_train.Cabin.isnull()].value_counts()
# df_cabin = pd.DataFrame({u'有':survived_cabin,u'无':survived_no_cabin}).transpose()
# df_cabin.plot(kind = 'bar',stacked = True)
# plt.title(u'按Cabin有无看获救情况')
# plt.ylabel(u'人数')
# plt.xlabel(u'Cabin有无')
# plt.show()

# survived_cabin = data_train.Survived[data_train.Cabin.notnull()].value_counts()
# survived_no_cabin = data_train.Survived[data_train.Cabin.isnull()].value_counts()
# df_cabin = pd.DataFrame({u'you':survived_cabin,u'wu':survived_no_cabin})
# df_cabin.plot(kind = 'bar',stacked = True)
# plt.title(u'按Cabin有无看获救情况')
# plt.ylabel(u'人数')
# plt.xlabel(u'Cabin有无')
# plt.show()


# age_df = data_train[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass','Cabin']]
#
# # 乘客分成已知年龄和未知年龄两部分
# known_age = age_df[age_df.Age.notnull()].as_matrix()
# unknown_age = age_df[age_df.Age.isnull()].as_matrix()
# print(known_age)
# # y即目标年龄
# y = known_age[:, 0]
#
# # X即特征属性值
# X = known_age[:, 1:]
#
# print(age_df[age_df.Cabin.notnull()])
# print(age_df[age_df.Cabin.isnull()])
#
#
# #将有cabin记录的标记为 Yes   没有的标记为   No
# age_df.loc[(age_df.Cabin.notnull()),'Cabin'] = 'Yes'
# age_df.loc[(age_df.Cabin.isnull()),'Cabin'] = 'No'
#
#
#
# print(age_df[age_df.Cabin == 'Yes'])
# print(age_df[age_df.Cabin == 'No'])
# print(os.getcwd())
def set_missing_ages(df):
    age_df = df[['Age','Fare','Parch','SibSp','Pclass']]
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    y = known_age[:,0]
    X = known_age[:,1:]

    rfr = RandomForestRegressor(random_state= 0,n_estimators= 200,n_jobs= -1)
    rfr.fit(X,y)
    predictedAges = rfr.predict(unknown_age[:,1:])
    df.loc[(df.Age.isnull()),'Age'] = predictedAges
    return df,rfr

def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()),'Cabin'] = 'Yse'
    df.loc[(df.Cabin.isnull()),'Cabin'] = 'No'
    return df


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
    # print(data_train)
    data_train, rfr = set_missing_ages(data_train)
    data_train = set_Cabin_type(data_train)
    # print(rfr)
    # print(data_train)

    dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
    dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
    dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
    dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')
    df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Pclass, dummies_Sex], axis=1)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    # print(df)
    # a = df['Age']
    # print('a.mean ={}  a.std = {} a.var = {}'.format(a.mean,a.std,a.var))
    scaler = preprocessing.StandardScaler()
    # age_scale_param = scaler.fit(df['Age'])
    df['Age_scaled'] = scaler.fit_transform(df['Age'])
    # fare_scale_param = scaler.fit(df['Fare'])
    df['Fare_scaled'] = scaler.fit_transform(df['Fare'])
    train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin.*|Embarked*|Sex*|Pclass*')
    print('the len of traindf is {}'.format(len(train_df.columns.values)))
    train_np = train_df.as_matrix()
    print('train_df = ')
    # print(train_df)
    print(train_df.columns.values)
    print(len(train_df.columns.values))

    #同样处理测试集
    data_test = pd.read_csv("D:/titanic/test.csv")
    data_test.loc[(data_test.Fare.isnull()), 'Fare'] = 0
    tmp_df = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    null_age = tmp_df[tmp_df.Age.isnull()].as_matrix()
    X = null_age[:, 1:]
    predictAges = rfr.predict(X)
    data_test.loc[(data_test.Age.isnull()), 'Age'] = predictAges

    data_test = set_Cabin_type(data_test)
    dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')
    dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix='Embarked')
    dummies_Sex = pd.get_dummies(data_test['Sex'], prefix='Sex')
    dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix='Pclass')
    df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Pclass, dummies_Sex], axis=1)
    df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    # print(df_test.columns)
    df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'])
    df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'])
    test_df = df_test.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin.*|Embarked*|Sex*|Pclass*')
    print('test_df.columns.values = ')
    print(test_df.columns.values)
    print(len(test_df.columns.values))
    test_np = test_df.as_matrix()
    # print(test_df.as_matrix())

    # #使用对数几率回归分类器
    # X = train_np[:, 1:]
    # y = train_np[:, 0]
    #
    # lr = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    # lr.fit(X, y)
    # print(lr)
    #
    # ## 使用对数几率 回归 预测
    # predictions = lr.predict(test_df)

    # #使用sklearn 的bagging策略来防止过拟合训练
    # bagging_X = train_np[:,1:]
    # bagging_y = train_np[:,0]
    # # print(bagging_X)
    # # print(bagging_y)
    # base_lr = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    # bagging_lr = BaggingRegressor(base_lr, n_estimators=10, max_samples=0.8, max_features=1.0, bootstrap=True,
    #                               bootstrap_features=False, n_jobs=-1)
    # bagging_lr.fit(bagging_X,bagging_y)
    # print(bagging_lr)

    # #使用svm
    # svc = SVC()
    # all_data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin.*|Embarked*|Sex*|Pclass*')
    # svc_X = all_data.as_matrix()[:, 1:]
    # svc_y = all_data.as_matrix()[:, 0]
    # svc.fit(svc_X,svc_y)
    # print('the validation result of svm is:')
    # print(cross_validation.cross_val_score(svc,svc_X,svc_y))


    # # 简单看看打分情况
    # clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    # all_data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    # X = all_data.as_matrix()[:, 1:]
    # y = all_data.as_matrix()[:, 0]
    # print(cross_validation.cross_val_score(clf, X, y, cv=10))
    # plot_learning_curve(clf, u"学习曲线", X, y)


    #打包所有分类器:对数几率回归、支持向量机、最近邻、决策树、随机森林、gbdt、xgboost
    lr = LogisticRegression()
    svc = SVC()
    knn = KNeighborsClassifier(n_neighbors=3)
    dt = DecisionTreeClassifier()
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=4, class_weight={0: 0.745, 1: 0.255})
    gbdt = GradientBoostingClassifier(n_estimators=500, learning_rate=0.03, max_depth=3)
    xgb = XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)


    # 打包所有分类器:对数几率回归、支持向量机、最近邻、决策树、随机森林、gbdt、xgboost
    lr = LogisticRegression()
    svc = SVC()
    knn = KNeighborsClassifier(n_neighbors=3)
    dt = DecisionTreeClassifier()
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=4, class_weight={0: 0.745, 1: 0.255})
    gbdt = GradientBoostingClassifier(n_estimators=500, learning_rate=0.03, max_depth=3)
    xgb = XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
    estimators = [('lr',lr),('svc',svc),('knn',knn),('dt',dt),('rf',rf),('gbdt',gbdt)]
    X = train_np[:, 1:]
    y = train_np[:, 0]
    for i in estimators:
        i[1].fit(X,y)
        print('the validation result of {} is:'.format(i[0]))
        print(cross_validation.cross_val_score(i[1],X,y,cv = 5))
    my_predictions = np.array([i[1].predict(X) for i in estimators]).T
    print(my_predictions.shape)
    print(my_predictions)




    # #预测并打印结果到csv
    # for i in estimators:
    #     predictions = i[1].predict(test_np)
        # result = pd.DataFrame(
        #     {'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
        # result.to_csv('D:/titanic/results/' +i[0] +'_predictions' + time.strftime('%Y-%m-%d %H-%M-%S',
        #                                                                            time.localtime(time.time())) + '.csv',
        #               index=False)


