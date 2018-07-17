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


data_train = pd.read_csv("D:/titanic/train.csv")
data_test = pd.read_csv("D:/titanic/test.csv")
# print(data_train.Sex.value_counts())
# print('***************')
# print(pd.DataFrame(data_train.groupby(['Sex','Survived']).count()['PassengerId']))
# print(pd.DataFrame(data_train.groupby(['Sex']).mean()['Survived']))
# print(data_train[data_train.Sex == 'male'].Survived.value_counts())
# print('*******************************************************')


# df = pd.DataFrame({'key1':list('aabba'),
#                   'key2': ['one','two','one','two','one'],
#                   'data1': np.random.randn(5),
#                   'data2': np.random.randn(5)})
# print(df)
# g = df.groupby(['key1','key2'])
# print(pd.DataFrame(g.count()))

# data_train.Age.value_counts().plot(kind = 'kde')
# plt.show()


# my_data = data_train.iloc[:10]
# my_data['PassengerId'].plot()
# my_data['Survived'].plot()
# plt.legend(loc = 'best')
# plt.show()


from sklearn.metrics import roc_curve, auc
f = open("D:/titanic/ROC_test.csv")
mydata = pd.read_csv(f)
print(mydata)
x1,y1= mydata['x1'],mydata['y1']
x2,y2= mydata['x2'],mydata['y2']
x3,y3= mydata['x3'],mydata['y3']

plt.plot(x1,y1,color ='green',label = 'Affinity propagation',linestyle ='-')
plt.plot(x2,y2,color ='red',label = 'k-means',linestyle ='--')
plt.plot(x3,y3,color ='blue',label = 'Random',linestyle ='-.')

plt.xlim((0,5.0))
plt.ylim((0,50))
plt.xlabel('False positive rate(%)')
plt.ylabel('True positive rate(%)')
plt.legend(loc = 'upper left')
plt.show()


# data_train.Age[data_train.Survived == 0].plot(kind = 'kde')
# data_train.Age[data_train.Survived == 1].plot(kind = 'kde')
# plt.legend((u'1',u'0'),loc = 'best')
# plt.show()
# #
# data_train.Fare[data_train.Survived == 0].plot(kind = 'kde')
# data_train.Fare[data_train.Survived == 1].plot(kind = 'kde')
# plt.legend((u'0',u'1'),loc = 'best')
# plt.gca().set_xticklabels(('0','50','100','150','200','250'))
# plt.show()
#
# data_train.Age[data_train.Embarked == 'C'].plot(kind = 'kde')
# data_train.Age[data_train.Embarked == 'Q'].plot(kind = 'kde')
# data_train.Age[data_train.Embarked == 'S'].plot(kind = 'kde')
# plt.legend((u'C',u'Q',u'S'),loc = 'best')
# plt.show()

# for i in ['C','Q','S']:
#     for j in range(1,4,1):
#         data_train.Survived[data_train.Embarked == i][data_train.Pclass == j].value_counts().plot(kind = 'bar',label = '{}/{}'.format(i,j))
#
# plt.show()

# fig=plt.figure()
# fig.set(alpha=0.65) # 设置图像透明度，无所谓
# plt.title(u"根据舱等级和性别的获救情况")
#
# ax1=fig.add_subplot(141)
# data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar', label="female highclass", color='#FA2479')
# print(data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts())
# ax1.set_xticklabels([u"获救", u"未获救"], rotation=0)
# ax1.legend([u"女性/高级舱"], loc='best')
#
# ax2=fig.add_subplot(142, sharey=ax1)
# data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')
# print(data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts())
# ax2.set_xticklabels([u"未获救", u"获救"], rotation=0)
# plt.legend([u"女性/低级舱"], loc='best')
#
# ax3=fig.add_subplot(143, sharey=ax1)
# data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')
# print(data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts())
# ax3.set_xticklabels([u"未获救", u"获救"], rotation=0)
# plt.legend([u"男性/高级舱"], loc='best')
#
# ax4=fig.add_subplot(144, sharey=ax1)
# data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')
# print(data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts())
# ax4.set_xticklabels([u"未获救", u"获救"], rotation=0)
# plt.legend([u"男性/低级舱"], loc='best')
# print(type(data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts()))
# plt.show()

#
# survived_1 = data_train.SibSp[data_train.Survived == 1].value_counts()
# survived_0 = data_train.SibSp[data_train.Survived == 0].value_counts()
# df = pd.DataFrame({u'1':survived_1,u'0':survived_0})
# df.plot(kind = 'bar',stacked = True)
# plt.show()


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

#
# if __name__ == '__main__':
#     # 将Embarked 为空的赋值为众数 'S'
#     # s = data_train.Embarked.isnull()
#     # print(data_train.Embarked)
#     # data_train['Embarked'] = data_train['Embarked'].apply(lambda x:x if x == 'S' or x == 'Q' or x == 'C' else 'S')
#     # print(data_train[s])
#     # print(data_train.Embarked)
#     data_train.Embarked.fillna('S', inplace=True)
#
#     # # 将Cabin 为 空/非空 作为一个类别特征
#     s = data_train.Cabin.isnull()
#     data_train['Cabin'] = data_train['Cabin'].isnull().apply(lambda x: 'Null' if x is True else'Not Null')
#     data_test['Cabin'] = data_test['Cabin'].isnull().apply(lambda x: 'Null' if x is True else'Not Null')
#     # del data_train['Name'],data_test['Name']
#     del data_train['Ticket'], data_test['Ticket']
#
#
#     # 将年龄离散化处理，并且处理缺省值
#     # 1. 为空的归为一类             2.分类的按年龄离散化
#     # 以5岁为一个周期离散，并且10以下，60以上的分别归类
#     def Age_map(x):
#         if x < 10:
#             return '10-'
#         if x < 60:
#             return '{}-{}'.format(int(x // 5 * 5), int(x // 5 * 5 + 5))
#         elif x >= 60:
#             return '60+'
#         else:
#             return 'Null'
#
#     data_train['Age_map'] = data_train['Age'].apply(lambda x:Age_map(x))
#     data_test['Age_map'] = data_test['Age'].apply(lambda x:Age_map(x))
#     del data_train['Age'],data_test['Age']
#     print(data_train.groupby(['Age_map']).agg(['count','mean'])['Survived'])
#
#
#     # print(data_test[data_test.Fare.isnull()])
#     #将data_test的 Fare缺省值取均值
#     data_test.loc[data_test.Fare.isnull(),'Fare'] = data_test[(data_test.Pclass == 3)&(data_test.Embarked == 'S')&
#     (data_test.Sex == 'male')].dropna().Fare.mean()
#
#     # #数据中Fare分布太宽，做一下scaling，加速模型收敛速度
#     scaler = preprocessing.StandardScaler()
#     data_train['Fare_Scaled'] = scaler.fit_transform(data_train['Fare'])
#     data_test['Fare_Scaled'] = scaler.fit_transform(data_test['Fare'])
#     del data_train['Fare'],data_test['Fare']
#     # print(data_train)
#     train_x = pd.concat(
#         [data_train[['SibSp', 'Parch', 'Fare_Scaled']], pd.get_dummies(data_train[['Pclass', 'Sex', 'Cabin', 'Embarked', 'Age_map']])],
#         axis=1)
#     train_y = data_train.Survived
#     test_x = pd.concat(
#         [data_test[['SibSp', 'Parch', 'Fare_Scaled']],
#          pd.get_dummies(data_test[['Pclass', 'Sex', 'Cabin', 'Embarked', 'Age_map']])],
#         axis=1)
#
#
#
#     #对于一个基分类器，可以使用GridSearchCV来自动调参
#     # base_line_model = LogisticRegression()
#     # param = {'penalty':['l1','l2'],'C':[0.1,0.5,1.0,5.0]}
#     # grd = GridSearchCV(estimator = base_line_model,param_grid = param,cv = 5,n_jobs = 3)
#     # grd.fit(train_x,train_y)
#     # print(grd.best_estimator_)
#
#
#     # plot_learning_curve(grd.best_estimator_,u'learning_rate',train_x,train_y)     #打印学习曲线
#
#
#
#     # 打包所有分类器:对数几率回归、支持向量机、最近邻、决策树、随机森林、gbdt、xgboost
#     lr = LogisticRegression()
#     svc = SVC()
#     knn = KNeighborsClassifier(n_neighbors=3)
#     dt = DecisionTreeClassifier()
#     rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=4, class_weight={0: 0.745, 1: 0.255})
#     gbdt = GradientBoostingClassifier(n_estimators=500, learning_rate=0.03, max_depth=3)
#     xgb = XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
#     estimators = [('lr', lr), ('svc', svc), ('knn', knn), ('dt', dt), ('rf', rf), ('gbdt', gbdt), ('xgb', xgb)]
#
#     print('train_x =')
#     print(train_x)
#     print(test_x)
#     print(train_x.columns.values)
#     print(test_x.columns.values)
#
#     for i in estimators:
#         i[1].fit(train_x, train_y)
#         print('the validation result of {} is:'.format(i[0]))
#         # print(cross_validation.cross_val_score(i[1], train_x, train_y, cv=5))
#         cross_result = list(cross_validation.cross_val_score(i[1], train_x, train_y, cv=5))
#         cross_result.append(np.mean(cross_result))
#         print(cross_result)
#
#
#
#
#     # 预测并打印结果到csv
#     for i in estimators:
#         predictions = i[1].predict(test_x)
#         # result = pd.DataFrame(
#         #     {'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
#         # result.to_csv('D:/titanic/results/' +i[0] +'_predictions' + time.strftime('%Y-%m-%d %H-%M-%S',
#         #                                                                            time.localtime(time.time())) + '.csv',
#         #               index=False)
#
#
