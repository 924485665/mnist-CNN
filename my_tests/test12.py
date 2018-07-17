#coding:utf-8
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
import pandas as pd  # 数据分析
import numpy as np  # 科学计算
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
from sklearn.learning_curve import learning_curve
from sklearn.ensemble import BaggingRegressor
from sklearn import cross_validation
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve,KFold
import time
import os
from sklearn.feature_extraction.text import TfidfTransformer

# warnings.simplefilter('ignore', DeprecationWarning)
import warnings
warnings.filterwarnings('ignore')
def fxn():
    warnings.warn("deprecated", DeprecationWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


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
#
# data_train.Age[data_train.Survived == 0].plot(kind = 'kde')
# data_train.Age[data_train.Survived == 1].plot(kind = 'kde')
# plt.legend((u'1',u'0'),loc = 'best')
# plt.show()
#
# data_train.Fare[data_train.Survived == 0].plot(kind = 'kde')
# data_train.Fare[data_train.Survived == 1].plot(kind = 'kde')
# plt.legend((u'0',u'1'),loc = 'best')
# plt.gca().set_xticklabels(('0','50','100','150','200','250'))
# plt.show()

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



#知乎的stacking方法，但没有使用5-折交叉训练，所以效果略差
class Ensemble(object):
    def __init__(self, estimators):
        self.estimator_names = []
        self.estimators = []
        for i in estimators:
            self.estimator_names.append(i[0])
            self.estimators.append(i[1])
        self.clf = LogisticRegression()

    def fit(self, train_x, train_y):
        for i in self.estimators:
            i.fit(train_x, train_y)
        x = np.array([i.predict(train_x) for i in self.estimators]).T
        y = train_y
        self.clf.fit(x, y)

    def predict(self, x):
        x = np.array([i.predict(x) for i in self.estimators]).T
        # print(x)
        return self.clf.predict(x)

    def score(self, x, y):
        s = precision_score(y, self.predict(x))
        return s

if __name__ =='__main__':
    combine_df = pd.concat([data_train,data_test])
    #Name特征
    combine_df['Name_Len'] = combine_df.Name.apply(lambda x:len(x))
    combine_df['Name_Len'] = pd.qcut(combine_df.Name_Len,5)
    df_name = pd.get_dummies(combine_df.Name_Len, prefix='Name_Len')
    combine_df = pd.concat([combine_df, df_name], axis=1).drop('Name_Len', axis=1)

    #不同的称谓类似，有显著不同的获救概率
    # combine_df.groupby(combine_df['Name'].apply(lambda x: x.split(', ')[1]).apply(lambda x: x.split('.')[0]))[
    #     'Survived'].mean().plot(kind = 'bar')
    # plt.show()

    combine_df['Title'] = combine_df['Name'].apply(lambda x: x.split(', ')[1]).apply(lambda x: x.split('.')[0])
    combine_df['Title'] = combine_df['Title'].replace(
        ['Don', 'Dona', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col', 'Sir', 'Dr'], 'Mr')
    combine_df['Title'] = combine_df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    combine_df['Title'] = combine_df['Title'].replace(['the Countess', 'Mme', 'Lady', 'Dr'], 'Mrs')
    df = pd.get_dummies(combine_df['Title'], prefix='Title')
    combine_df = pd.concat([combine_df, df], axis=1)
    # print(combine_df.Title)


    #同一个family下的生存死亡模式有很大程度上是相同的，例如：有一个family有一个女性死亡，这个family其他的女性的死亡概率也比较高。
    # 因此，我们标注出这些特殊的family即可
    combine_df['Fname'] = combine_df['Name'].apply(lambda x: x.split(',')[0])
    combine_df['Familysize'] = combine_df['SibSp'] + combine_df['Parch']
    dead_female_Fname = list(set(combine_df[(combine_df.Sex == 'female') & (combine_df.Age >= 12)
                                            & (combine_df.Survived == 0) & (combine_df.Familysize > 1)][
                                     'Fname'].values))
    survive_male_Fname = list(set(combine_df[(combine_df.Sex == 'male') & (combine_df.Age >= 12)
                                             & (combine_df.Survived == 1) & (combine_df.Familysize > 1)][
                                      'Fname'].values))
    combine_df['Dead_female_family'] = np.where(combine_df['Fname'].isin(dead_female_Fname), 1, 0)
    combine_df['Survive_male_family'] = np.where(combine_df['Fname'].isin(survive_male_Fname), 1, 0)
    combine_df = combine_df.drop(['Name', 'Fname'], axis=1)


    #Age  添加一个小孩子标签
    # print(combine_df.Age)
    group = combine_df.groupby(['Title', 'Pclass'])['Age']
    # print(group.count())
    print('******************************************')
    combine_df['Age'] = group.transform(lambda x: x.fillna(x.median()))

    combine_df = combine_df.drop('Title', axis=1)

    combine_df['IsChild'] = np.where(combine_df['Age'] <= 12, 1, 0)
    # print(combine_df.columns)
    combine_df['Age'] = pd.cut(combine_df['Age'], 5)
    combine_df = combine_df.drop('Age', axis=1)

    #Familysize    我们将上面提取过的Familysize再离散化
    combine_df['Familysize'] = np.where(combine_df['Familysize'] == 0, 'solo',
                                        np.where(combine_df['Familysize'] <= 3, 'normal', 'big'))
    df = pd.get_dummies(combine_df['Familysize'], prefix='Familysize')
    combine_df = pd.concat([combine_df, df], axis=1).drop(['SibSp', 'Parch', 'Familysize'],axis = 1)
    # print(combine_df.columns)
    # print(len(combine_df.columns))


    #Ticket
    combine_df['Ticket_Lett'] = combine_df['Ticket'].apply(lambda x: str(x)[0])
    combine_df['Ticket_Lett'] = combine_df['Ticket_Lett'].apply(lambda x: str(x))

    combine_df['High_Survival_Ticket'] = np.where(combine_df['Ticket_Lett'].isin(['1', '2', 'P']), 1, 0)
    combine_df['Low_Survival_Ticket'] = np.where(combine_df['Ticket_Lett'].isin(['A', 'W', '3', '7']), 1, 0)
    combine_df = combine_df.drop(['Ticket', 'Ticket_Lett'], axis=1)

    #Embarked
    combine_df.Embarked = combine_df.Embarked.fillna('S')
    df = pd.get_dummies(combine_df['Embarked'], prefix='Embarked')
    combine_df = pd.concat([combine_df, df], axis=1).drop('Embarked', axis=1)

    #Cabin
    combine_df['Cabin_isNull'] = np.where(combine_df['Cabin'].isnull(), 0, 1)
    combine_df = combine_df.drop('Cabin', axis=1)


    #Pclass
    df = pd.get_dummies(combine_df['Pclass'],prefix = 'Pclass')
    combine_df = pd.concat([combine_df,df],axis = 1).drop('Pclass',axis = 1)

    #Sex
    df = pd.get_dummies(combine_df['Sex'], prefix='Sex')
    combine_df = pd.concat([combine_df, df], axis=1).drop('Sex', axis=1)

    #
    # #Fare  缺省值用众数填充，之后进行离散化
    # combine_df['Fare'] = pd.qcut(combine_df.Fare, 3)
    # # print(combine_df.Fare)
    # df = pd.get_dummies(combine_df.Fare, prefix='Fare').drop('Fare_(-0.001, 8.662]', axis=1)
    # combine_df = pd.concat([combine_df, df], axis=1).drop('Fare', axis=1)


    # Fare
    combine_df['Fare'].fillna(combine_df['Fare'].dropna().median(), inplace=True)
    combine_df['Low_Fare'] = np.where(combine_df['Fare'] <= 8.662, 1, 0)
    combine_df['High_Fare'] = np.where(combine_df['Fare'] >= 26, 1, 0)
    combine_df = combine_df.drop('Fare', axis=1)



    print(combine_df.columns.values)
    print(len(combine_df.columns.values))


    print('before LabelEncoder.....')
    print(combine_df)
    print(len(combine_df.columns))

    #所有特征转化成数值型编码
    features = combine_df.drop(["PassengerId", "Survived"], axis=1).columns
    le = LabelEncoder()
    for feature in features:
        le = le.fit(combine_df[feature])
        combine_df[feature] = le.transform(combine_df[feature])

    print('After LabelEncoder....')
    print(combine_df)

    # 防止xgboost 不识别特征中的  ']'
    combine_df.rename(columns=lambda x: x.replace('[', '').replace(']',''), inplace=True)

    #得到训练/测试数据
    X_all = combine_df.iloc[:891, :].drop(["PassengerId", "Survived"], axis=1)
    Y_all = combine_df.iloc[:891, :]["Survived"]
    X_test = combine_df.iloc[891:, :].drop(["PassengerId", "Survived"], axis=1)
    # print('X_all = ')
    # print(X_all)
    # print(Y_all)
    # print(X_test)
    # print(X_all.columns.values)
    # print(X_test.columns.values)
    print('*******************************************************')


    #
    # #分别考察逻辑回归、支持向量机、最近邻、决策树、随机森林、gbdt、xgbGBDT几类算法的性能。
    lr = LogisticRegression()
    svc = SVC()
    knn = KNeighborsClassifier(n_neighbors=3)
    dt = DecisionTreeClassifier()
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=4, class_weight={0: 0.745, 1: 0.255})
    gbdt = GradientBoostingClassifier(n_estimators=500, learning_rate=0.03, max_depth=3)
    xgb = XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
    clfs = [('lr', lr), ('svc', svc), ('knn', knn), ('dt', dt), ('rf', rf), ('gbdt', gbdt),('xgb',xgb)]
    #
    # kfold = 10
    # cv_results = []
    # for classifier in clfs:
    #     cv_results.append(cross_val_score(classifier, X_all, y=Y_all, scoring="accuracy", cv=kfold, n_jobs=4))

    # for classifier in clfs:
    #     print('the cross_value of {} is:'.format(classifier[0]))
    #     # print(cross_val_score(classifier[1], X_all, y=Y_all, scoring="accuracy", cv=kfold, n_jobs=4))
    #     cross_result = list(cross_validation.cross_val_score(classifier[1], X_all, Y_all, cv=5))
    #     cross_result.append(np.mean(cross_result))
    #     print(cross_result)





    #
    # #XGBClassifier 很坑，属性名中不能有 '[,]'
    # demo_train = X_all.iloc[:,:]           #是深拷贝，怎么实现浅拷贝？
    # demo_train['Fare_1'] = X_all['Fare_(8.662, 26.0]']
    # demo_train['Fare_2'] = X_all['Fare_(26.0, 512.329]']
    # demo_test = X_test.iloc[:,:]
    # demo_test['Fare_1'] = X_all['Fare_(8.662, 26.0]']
    # demo_test['Fare_2'] = X_all['Fare_(26.0, 512.329]']
    # demo_test = demo_test.drop(['Fare_(8.662, 26.0]', 'Fare_(26.0, 512.329]'], axis=1)
    # print(X_all.columns.values)
    # print(demo_train.columns.values)
    # demo_train = demo_train.drop(['Fare_(8.662, 26.0]','Fare_(26.0, 512.329]'],axis = 1)
    # print(demo_train.columns.values)
    # print(cross_val_score(rf, demo_train, y=Y_all, scoring="accuracy", cv=kfold, n_jobs=4))
    # rf.fit(demo_train,Y_all)
    # print(rf.feature_importances_)
    # print(type(rf.feature_importances_))
    # print(rf.feature_importances_.sum())

    #
    # #实现浅拷贝X_all
    # demo = X_all.drop(['Fare_(8.662, 26.0]','Fare_(26.0, 512.329]'],axis = 1)
    # print(demo.columns.values)
    # print(X_all.columns.values)
    # demo['Fare1'] = X_all['Fare_(8.662, 26.0]']
    # demo['Fare2'] = X_all['Fare_(26.0, 512.329]']
    # print(demo.columns.values)
    # print(X_all.columns.values)

    # cv_means = []
    # cv_std = []
    # for cv_result in cv_results:
    #     cv_means.append(cv_result.mean())
    #     cv_std.append(cv_result.std())

    # cv_res = pd.DataFrame({"CrossValMeans": cv_means, "CrossValerrors": cv_std,
    #                        "Algorithm": ["LR", "SVC", 'KNN', 'decision_tree', "random_forest", "GBDT", "xgbGBDT"]})

    # # g = sns.barplot("CrossValMeans", "Algorithm", data=cv_res, palette="Set3", orient="h", **{'xerr': cv_std})
    # # g.set_xlabel("Mean Accuracy")
    # # g = g.set_title("Cross validation scores")






    #观察发现不同的模型的feature importance 有比较大的差别，，，把他们组合再一起会不会更好呢



    #集成框架Ensemble，我们把基分类器丢进去。
    # bag = Ensemble([('xgb', xgb), ('lr', lr), ('rf', rf), ('svc', svc), ('gbdt', gbdt)])
    # bag = Ensemble([('xgb', xgb), ('lr', lr), ('gbdt', gbdt), ('rf', rf)])
    # score = 0
    # for i in range(0, 10):
    #     num_test = 0.20
    #     X_train, X_cv, Y_train, Y_cv = train_test_split(X_all, Y_all, test_size=num_test)
    #     bag.fit(X_train, Y_train)
    #     # Y_test = bag.predict(X_test)
    #     acc_xgb = round(bag.score(X_cv, Y_cv) * 100, 2)
    #     score += acc_xgb
    # print(score / 10)  # 0.8786
    #
    # print(X_all.values)

    #Out-of-Fold Predictions
    x = X_all.as_matrix()
    y = Y_all.as_matrix()
    test = X_test.as_matrix()
    # print(x,y)
    ntrain = x.shape[0]
    ntest = test.shape[0]
    kf = KFold(n_splits = 5)
    # print(ntrain,ntest,kf)

    def get_oof(clf,x_train,y_train,x_test):
        oof_train = np.zeros((ntrain,))
        oof_test = np.zeros((ntest,))
        oof_test_skf = np.empty((5,ntest))
        for i,(train_index,test_index) in enumerate(kf.split(x_train)):
            kf_x_train = x_train[train_index]
            kf_y_train = y_train[train_index]
            kf_x_test = x_train[test_index]

            clf.fit(kf_x_train,kf_y_train)

            oof_train[test_index] = clf.predict(kf_x_test)
            oof_test_skf[i,:] = clf.predict(x_test)

        oof_test = oof_test_skf.mean(axis = 0)
        # print('oof_train=')
        # print(oof_train.reshape(-1,1))
        # print(oof_train.reshape(-1,1)[0])
        # print(oof_train.reshape(-1,1)[:,0])
        # print('oof_test =')
        # print(oof_test.reshape(-1,1))
        return oof_train.reshape(-1,1)[:,0],oof_test.reshape(-1,1)[:,0]


    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    # print(oof_train.shape)
    # print(oof_test.shape)
    # print(type(oof_train))
    # oof_train,oof_test = get_oof(lr,x,y,test)
    # print(oof_train,oof_test)


    # # Random Forest parameters
    # rf_params = {
    #     'n_jobs': -1,
    #     'n_estimators': 500,
    #     'warm_start': True,
    #     # 'max_features': 0.2,
    #     'max_depth': 6,
    #     'min_samples_leaf': 2,
    #     'max_features': 'sqrt',
    #     'verbose': 0
    # }
    #
    # # Extra Trees Parameters
    # et_params = {
    #     'n_jobs': -1,
    #     'n_estimators': 500,
    #     # 'max_features': 0.5,
    #     'max_depth': 8,
    #     'min_samples_leaf': 2,
    #     'verbose': 0
    # }
    #
    # # AdaBoost parameters
    # ada_params = {
    #     'n_estimators': 500,
    #     'learning_rate': 0.75
    # }
    #
    # # Gradient Boosting parameters
    # gb_params = {
    #     'n_estimators': 500,
    #     # 'max_features': 0.2,
    #     'max_depth': 5,
    #     'min_samples_leaf': 2,
    #     'verbose': 0
    # }
    #
    # # Support Vector Classifier parameters
    # svc_params = {
    #     'kernel': 'linear',
    #     'C': 0.025
    # }


    # # Create 5 objects that represent our 4 models
    # rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=4, class_weight={0: 0.745, 1: 0.255})
    # et = ExtraTreesClassifier(n_jobs= -1,n_estimators=500,max_features= 0.5, max_depth= 8, min_samples_leaf= 2, verbose= 0)
    # ada = AdaBoostClassifier(n_estimators=500, learning_rate= 0.75)
    # gbdt = GradientBoostingClassifier(n_estimators= 500, max_features=0.2, max_depth=5, min_samples_leaf= 2, verbose= 0)
    # svc = SVC(kernel= 'linear',C= 0.025)

    stackings = [('lr', lr), ('svc', svc), ('knn', knn), ('dt', dt), ('rf', rf), ('gbdt', gbdt),('xgb',xgb)]
    # stackings = list([('xgb', xgb), ('lr', lr), ('gbdt', gbdt),('rf',rf),('svc',svc)])
    # stackings = [('et', et), ('gbdt', gbdt),('rf',rf),('svc',svc),('ada',ada)]
    middle_train = np.zeros((ntrain,len(stackings)))
    middle_test = np.zeros((ntest,len(stackings)))
    print(x)
    print(x.shape)
    print(y)
    print(y.shape)
    for i,clf in enumerate(stackings):
        middle_train[:,i],middle_test[:,i] = get_oof(clf[1],x,y,test)

    print('After first stacking level.......')
    print(middle_train)
    print(middle_train.shape)
    print(middle_test)
    print(middle_test.shape)
    print('Begin second level(use lr).......')
    second_model = LogisticRegression()
    param = {'penalty':['l1','l2'],'C':[0.1,0.5,5.0]}

    clf2 =GridSearchCV(estimator = second_model,param_grid = param,cv = 5,n_jobs = -1)
    # print(clf2)
    clf2.fit(middle_train,y)
    # print(clf2.best_estimator_)
    # print(clf2.best_params_)
    # print(clf2.best_score_)
    # print(clf2.best_index_)
    # print(clf2)





    # xgb = XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
    # knn = KNeighborsClassifier(n_neighbors=3)

    # print(combine_df.columns.values)

    # #use best estimator
    # my_lr = lr2.best_estimator_
    # my_lr.fit(middle_train,y)
    # predictions = my_lr.predict(middle_test)

    # lr2.fit(middle_train,y)
    # predictions = lr2.predict(middle_test)
    # print(predictsions)
    # print(predictsions.shape)
    # print(cross_validation.cross_val_score(lr2, middle_train, y, cv=5))


    clf2.best_estimator_.fit(middle_train,y)
    predictions = clf2.best_estimator_.predict(middle_test)
    print(predictions)
    print(predictions.shape)
    print(cross_validation.cross_val_score(xgb, middle_train, y, cv=5))


    #stacking saving
    result = pd.DataFrame(
        {'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
    result.to_csv('D:/titanic/results/' +'stacking'+'_predictions' + time.strftime('%Y-%m-%d %H-%M-%S',
                                                                               time.localtime(time.time())) + '.csv',
                  index=False)







    #
    # bag.fit(X_all,Y_all)
    # predictions = bag.predict(X_test)
    # result = pd.DataFrame(
    #     {'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
    # result.to_csv('D:/titanic/results/' +'bag' +'_predictions' + time.strftime('%Y-%m-%d %H-%M-%S',
    #                                                                            time.localtime(time.time())) + '.csv',
    #               index=False)
    #














    # 预测并打印结果到csv
    # for i in clfs:
        # i[1].fit(X_all,Y_all)
        # predictions = i[1].predict(X_test)
        # result = pd.DataFrame(
        #     {'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
        # result.to_csv('D:/titanic/results/' +i[0] +'_predictions' + time.strftime('%Y-%m-%d %H-%M-%S',
        #                                                                            time.localtime(time.time())) + '.csv',
        #               index=False)




