#coding:utf-8
# # import tensorflow as tf
# #
# # tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
# # FLAGS = tf.flags.FLAGS
# #
# #
# # class TextCNN(object):
# #     """
# #     A CNN for text classification.
# #     Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
# #     """
# #
# #     def __init__(
# #             self, sequence_length, num_classes, vocab_size,
# #             embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.8):
# #         # Placeholders for input, output and dropout
# #         self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
# #         self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
# #         self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
# #
# #         # Keeping track of l2 regularization loss (optional)
# #         l2_loss = tf.constant(0.0)
# #         print(l2_reg_lambda)
# #
# #     # def pri(self):
# #     #     print(sequence_length)
# #
# #
# # cnn = TextCNN(
# #     sequence_length=1,
# #     num_classes=1,
# #     vocab_size=1,
# #     embedding_size=1,
# #     filter_sizes=1,
# #     num_filters=1,
# #     )
# #
# # # cnn.pri()
#
# #coding:utf-8
# # import matplotlib.pyplot as plt
# # plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# # plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
# # import pandas as pd  # 数据分析
# # import numpy as np  # 科学计算
# # from pandas import Series, DataFrame
# # import matplotlib.pyplot as plt
# # from sklearn import linear_model
# # from sklearn.ensemble import RandomForestRegressor
# # import sklearn.preprocessing as preprocessing
# # from sklearn import linear_model
# # from sklearn.ensemble import BaggingRegressor
# # import time
# # from sklearn.preprocessing import LabelEncoder
# # from sklearn.model_selection import train_test_split
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.svm import SVC, LinearSVC
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.neighbors import KNeighborsClassifier
# # from sklearn.naive_bayes import GaussianNB
# # from sklearn.linear_model import Perceptron
# # from sklearn.linear_model import SGDClassifier
# # from sklearn.tree import DecisionTreeClassifier
# # from xgboost import XGBClassifier
# # from sklearn.metrics import precision_score
# # from sklearn.ensemble import GradientBoostingClassifier
# # from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve,KFold
# # import os
# # from sklearn.feature_extraction.text import TfidfTransformer
#
# # warnings.simplefilter('ignore', DeprecationWarning)
# # import warnings
# # warnings.filterwarnings('ignore')
# # def fxn():
# #     warnings.warn("deprecated", DeprecationWarning)
# # with warnings.catch_warnings():
# #     warnings.simplefilter("ignore")
# #     fxn()
# #
# #
# # data_train = pd.read_csv("D:/titanic/train.csv")
# # data_test = pd.read_csv("D:/titanic/test.csv")
# # p = os.path.abspath(os.path.join(os.path.dirname('E:/daima/ChemistryDataMining/Preprocess/__pycache__/'), os.path.par
#
# # X_all = pd.DataFrame({'col1':[1,2,3],'col2':[4,5,6]})
# # Y_all = pd.DataFrame(['1','0','0'])
# # print(X_all)
# # print(Y_all)
# # demo = pd.concat([X_all,Y_all],axis = 1)
# # print(demo)
# # num_test = 0.5
# # X_train, X_cv, Y_train, Y_cv = train_test_split(X_all, Y_all, test_size=num_test)
# # print(X_train)
#
# # a = np.array([[1,2,3],[4,5,6]])
# # b = np.array([[11,21,31],[7,8,9]])
# # print(a)
# # print(b)
# # print(a.shape)
# # c = np.concatenate((a,b),axis = 0)
# # d = np.concatenate((a,b),axis = 1)
# # print(c)
# # print(d)
# # print(type(d))
# # print(d[1])
# # print(d[:,1])
# # e=np.arange(9).reshape(3,3)
# # print(e)
# # print(type(e))
# # print(e[1])
# # print(e[:,1])
# # e[:,1] = [100,90,80]
# # print(e)
# # X_all = np.arange(20).reshape(5,4)
# # Y_all = np.arange(5)
# #
# #
# #
# # num_test = .2
# # X_train, X_cv, Y_train, Y_cv = train_test_split(X_all, Y_all, test_size=num_test)
# # print(X_all)
# # print(Y_all)
# # print(X_train)
# # print(X_cv)
# # print(Y_train)
# # print(Y_cv)
# #
# # combine_df = pd.concat([data_train,data_test])
# # print(combine_df.columns.values)
# # combine_df['Family_Size'] = combine_df['SibSp'] + combine_df['Parch'] + 1
# # print(combine_df)
# #
# # combine_df.groupby(['Family_Size']).mean()['Survived'].plot(kind = 'bar')
# #
# #
# # combine_df['IsAlone'] = np.where(combine_df.Family_Size == 1, 1 ,0)
# # print(combine_df[['Family_Size','IsAlone']])
# # print(combine_df.groupby(['IsAlone']).mean()['Survived'])
# #
# #
# # print(combine_df.Age)
# # mean = combine_df['Age'].mean()
# # std = combine_df['Age'].std()
# # counts  = combine_df.Age.isnull().sum()
# # random_int_list = np.random.randint(mean - std,mean + std,counts)
# # # combine_df[combine_df.Age.isnull()]['Age'] = random_int_list
# # # combine_df.loc[(combine_df.Age.isnull()),'Age'] = np.random.randint(mean - std,mean + std,counts)
# # # print(combine_df.Age)
# # temp_df = data_train.Age[data_train.Pclass == 1 ][ data_train.SibSp == 0]
# # print(temp_df)
# # data_train['Age'] = data_train.groupby(['Pclass','SibSp'])['Age'].transform(lambda x:x.fillna(x.median()))
# # print(data_train.Age[data_train.Pclass == 1 ][ data_train.SibSp == 0])
# #
# #
# #
# # print(combine_df.Sex)
# # combine_df['Sex'] = combine_df.Sex.map({'male':1,'female':0}).astype(int)
# # print(combine_df.Sex)
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# # class SklearnHelper(object):
# #     def __init__(self, clf, seed=0, params=None):
# #         params['random_state'] = seed
# #         self.clf = clf(**params)
# #         print(**params)
# #
# #
# # rf_params = {
# #     'n_jobs': -1,
# #     'n_estimators': 500,
# #      'warm_start': True,
# #      #'max_features': 0.2,
# #     # 'max_depth': 6,
# #     # 'min_samples_leaf': 2,
# #     # 'max_features' : 'sqrt',
# #     # 'verbose': 0
# # }
# # rf = SklearnHelper(clf=RandomForestClassifier, seed=0, params=rf_params)
# #
#
#
#
# #coding:utf-8
# # import matplotlib.pyplot as plt
# # plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# # plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
# # import pandas as pd  # 数据分析
# # import numpy as np  # 科学计算
# # from pandas import Series, DataFrame
# # import matplotlib.pyplot as plt
# # from sklearn import linear_model
# # from sklearn.ensemble import RandomForestRegressor
# # import sklearn.preprocessing as preprocessing
# # from sklearn import linear_model
# # from sklearn.ensemble import BaggingRegressor
# # from sklearn import cross_validation
# # import time
# # from sklearn.preprocessing import LabelEncoder
# # from sklearn.model_selection import train_test_split
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.svm import SVC, LinearSVC
# # from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier
# # from sklearn.neighbors import KNeighborsClassifier
# # from sklearn.naive_bayes import GaussianNB
# # from sklearn.linear_model import Perceptron
# # from sklearn.linear_model import SGDClassifier
# # from sklearn.tree import DecisionTreeClassifier
# # from xgboost import XGBClassifier
# # from sklearn.metrics import precision_score
# # from sklearn.ensemble import GradientBoostingClassifier
# # from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve,KFold
# # import time
# # import os
# # from sklearn.feature_extraction.text import TfidfTransformer
# #
# # # warnings.simplefilter('ignore', DeprecationWarning)
# # import warnings
# # warnings.filterwarnings('ignore')
# # def fxn():
# #     warnings.warn("deprecated", DeprecationWarning)
# # with warnings.catch_warnings():
# #     warnings.simplefilter("ignore")
# #     fxn()
# #
# #
# # data_train = pd.read_csv("D:/titanic/train.csv")
# # data_test = pd.read_csv("D:/titanic/test.csv")
# # print(data_train.Sex.value_counts())
# # print('***************')
# # print(pd.DataFrame(data_train.groupby(['Sex','Survived']).count()['PassengerId']))
# # print(pd.DataFrame(data_train.groupby(['Sex']).mean()['Survived']))
# # print(data_train[data_train.Sex == 'male'].Survived.value_counts())
# # print('*******************************************************')
#
#
# # df = pd.DataFrame({'key1':list('aabba'),
# #                   'key2': ['one','two','one','two','one'],
# #                   'data1': np.random.randn(5),
# #                   'data2': np.random.randn(5)})
# # print(df)
# # g = df.groupby(['key1','key2'])
# # print(pd.DataFrame(g.count()))
#
# # data_train.Age.value_counts().plot(kind = 'kde')
# # plt.show()
# #
# # data_train.Age[data_train.Survived == 0].plot(kind = 'kde')
# # data_train.Age[data_train.Survived == 1].plot(kind = 'kde')
# # plt.legend((u'1',u'0'),loc = 'best')
# # plt.show()
# #
# # data_train.Fare[data_train.Survived == 0].plot(kind = 'kde')
# # data_train.Fare[data_train.Survived == 1].plot(kind = 'kde')
# # plt.legend((u'0',u'1'),loc = 'best')
# # plt.gca().set_xticklabels(('0','50','100','150','200','250'))
# # plt.show()
#
# # data_train.Age[data_train.Embarked == 'C'].plot(kind = 'kde')
# # data_train.Age[data_train.Embarked == 'Q'].plot(kind = 'kde')
# # data_train.Age[data_train.Embarked == 'S'].plot(kind = 'kde')
# # plt.legend((u'C',u'Q',u'S'),loc = 'best')
# # plt.show()
#
# # for i in ['C','Q','S']:
# #     for j in range(1,4,1):
# #         data_train.Survived[data_train.Embarked == i][data_train.Pclass == j].value_counts().plot(kind = 'bar',label = '{}/{}'.format(i,j))
# #
# # plt.show()
#
# # fig=plt.figure()
# # fig.set(alpha=0.65) # 设置图像透明度，无所谓
# # plt.title(u"根据舱等级和性别的获救情况")
# #
# # ax1=fig.add_subplot(141)
# # data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar', label="female highclass", color='#FA2479')
# # print(data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts())
# # ax1.set_xticklabels([u"获救", u"未获救"], rotation=0)
# # ax1.legend([u"女性/高级舱"], loc='best')
# #
# # ax2=fig.add_subplot(142, sharey=ax1)
# # data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')
# # print(data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts())
# # ax2.set_xticklabels([u"未获救", u"获救"], rotation=0)
# # plt.legend([u"女性/低级舱"], loc='best')
# #
# # ax3=fig.add_subplot(143, sharey=ax1)
# # data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')
# # print(data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts())
# # ax3.set_xticklabels([u"未获救", u"获救"], rotation=0)
# # plt.legend([u"男性/高级舱"], loc='best')
# #
# # ax4=fig.add_subplot(144, sharey=ax1)
# # data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')
# # print(data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts())
# # ax4.set_xticklabels([u"未获救", u"获救"], rotation=0)
# # plt.legend([u"男性/低级舱"], loc='best')
# # print(type(data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts()))
# # plt.show()
#
# #
# # survived_1 = data_train.SibSp[data_train.Survived == 1].value_counts()
# # survived_0 = data_train.SibSp[data_train.Survived == 0].value_counts()
# # df = pd.DataFrame({u'1':survived_1,u'0':survived_0})
# # df.plot(kind = 'bar',stacked = True)
# # plt.show()
#
#
#
#
# # def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
# #                         train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
# #     """
# #     画出data在某模型上的learning curve.
# #     参数解释
# #     ----------
# #     estimator : 你用的分类器。
# #     title : 表格的标题。
# #     X : 输入的feature，numpy类型
# #     y : 输入的target vector
# #     ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
# #     cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
# #     n_jobs : 并行的的任务数(默认1)
# #     """
# #     train_sizes, train_scores, test_scores = learning_curve(
# #         estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)
# #
# #     train_scores_mean = np.mean(train_scores, axis=1)
# #     train_scores_std = np.std(train_scores, axis=1)
# #     test_scores_mean = np.mean(test_scores, axis=1)
# #     test_scores_std = np.std(test_scores, axis=1)
# #
# #     if plot:
# #         plt.figure()
# #         plt.title(title)
# #         if ylim is not None:
# #             plt.ylim(*ylim)
# #         plt.xlabel(u"训练样本数")
# #         plt.ylabel(u"得分")
# #         plt.gca().invert_yaxis()
# #         plt.grid()
# #
# #         plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
# #                          alpha=0.1, color="b")
# #         plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
# #                          alpha=0.1, color="r")
# #         plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
# #         plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"交叉验证集上得分")
# #
# #         plt.legend(loc="best")
# #
# #         plt.draw()
# #         plt.show()
# #         plt.gca().invert_yaxis()
# #
# #     midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
# #     diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
# #     return midpoint, diff
#
#
#
# # #知乎的stacking方法，但没有使用5-折交叉训练，所以效果略差
# # class Ensemble(object):
# #     def __init__(self, estimators):
# #         self.estimator_names = []
# #         self.estimators = []
# #         for i in estimators:
# #             self.estimator_names.append(i[0])
# #             self.estimators.append(i[1])
# #         self.clf = LogisticRegression()
# #
# #     def fit(self, train_x, train_y):
# #         for i in self.estimators:
# #             i.fit(train_x, train_y)
# #         x = np.array([i.predict(train_x) for i in self.estimators]).T
# #         y = train_y
# #         self.clf.fit(x, y)
# #
# #     def predict(self, x):
# #         x = np.array([i.predict(x) for i in self.estimators]).T
# #         # print(x)
# #         return self.clf.predict(x)
# #
# #     def score(self, x, y):
# #         s = precision_score(y, self.predict(x))
# #         return s
#
# #使用了k-折 交叉训练
# class Stacking(object):
#     def __init__(self, first_estimators):
#         self.estimators = []
#         self.first_esti_nums = len(first_estimators)
#         for i in first_estimators:
#             self.estimators.append(i[1])
#
#
#     def get_oof_predictions(self,clf,train_x,train_y,test,Kfold):
#         ntrain = train_x.shape[0]
#         ntest = test.shape[0]
#         oof_train = np.zeros((ntrain,))
#         oof_test = np.zeros((ntest,))
#         oof_test_skf = np.empty((Kfold, ntest))
#         kf = KFold(n_splits=Kfold)
#         for i, (train_index, test_index) in enumerate(kf.split(train_x)):
#             kf_x_train = train_x[train_index]
#             kf_y_train = train_y[train_index]
#             kf_x_test = train_x[test_index]
#
#             clf.fit(kf_x_train, kf_y_train)
#
#             oof_train[test_index] = clf.predict(kf_x_test)
#             oof_test_skf[i, :] = clf.predict(test)
#
#         oof_test = oof_test_skf.mean(axis = 0)
#         return oof_train.reshape(-1,1)[:,0],oof_test.reshape(-1,1)[:,0]
#
#
#
#
#     def fit_predict(self, train_x, train_y,test,Kfold):
#         # print('Kfold = {}'.format(Kfold))
#         ntrain = train_x.shape[0]
#         ntest = test.shape[0]
#         self.middle_train = np.zeros((ntrain, self.first_esti_nums))
#         self.middle_test = np.zeros((ntest, self.first_esti_nums))
#         for i, clf in enumerate(self.estimators):
#             self.middle_train[:, i], self.middle_test[:, i] = self.get_oof_predictions(clf, train_x, train_y, test,Kfold)
#         second_estimator = GridSearchCV(estimator=LogisticRegression(), param_grid=param_lr, cv=5, n_jobs=-1)
#         second_estimator.fit(self.middle_train,train_y)
#         second_estimator.best_estimator_.fit(self.middle_train, train_y)
#         predictions = second_estimator.best_estimator_.predict(self.middle_test)
#         return predictions
#
#     def score(self, x, y):
#         s = precision_score(y,self.fit_predict(x,y,x,10))
#         return s
#
# if __name__ =='__main__':
#     combine_df = pd.concat([data_train,data_test])
#     #Name特征
#     combine_df['Name_Len'] = combine_df.Name.apply(lambda x:len(x))
#     combine_df['Name_Len'] = pd.qcut(combine_df.Name_Len,5)
#     df_name = pd.get_dummies(combine_df.Name_Len, prefix='Name_Len')
#     combine_df = pd.concat([combine_df, df_name], axis=1).drop('Name_Len', axis=1)
#
#     #不同的称谓类似，有显著不同的获救概率
#     # combine_df.groupby(combine_df['Name'].apply(lambda x: x.split(', ')[1]).apply(lambda x: x.split('.')[0]))[
#     #     'Survived'].mean().plot(kind = 'bar')
#     # plt.show()
#
#     combine_df['Title'] = combine_df['Name'].apply(lambda x: x.split(', ')[1]).apply(lambda x: x.split('.')[0])
#     combine_df['Title'] = combine_df['Title'].replace(
#         ['Don', 'Dona', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col', 'Sir', 'Dr'], 'Mr')
#     combine_df['Title'] = combine_df['Title'].replace(['Mlle', 'Ms'], 'Miss')
#     combine_df['Title'] = combine_df['Title'].replace(['the Countess', 'Mme', 'Lady', 'Dr'], 'Mrs')
#     df = pd.get_dummies(combine_df['Title'], prefix='Title')
#     combine_df = pd.concat([combine_df, df], axis=1)
#     # print(combine_df.Title)
#
#
#     #同一个family下的生存死亡模式有很大程度上是相同的，例如：有一个family有一个女性死亡，这个family其他的女性的死亡概率也比较高。
#     # 因此，我们标注出这些特殊的family即可
#     combine_df['Fname'] = combine_df['Name'].apply(lambda x: x.split(',')[0])
#     combine_df['Familysize'] = combine_df['SibSp'] + combine_df['Parch']
#     dead_female_Fname = list(set(combine_df[(combine_df.Sex == 'female') & (combine_df.Age >= 12)
#                                             & (combine_df.Survived == 0) & (combine_df.Familysize > 1)][
#                                      'Fname'].values))
#     survive_male_Fname = list(set(combine_df[(combine_df.Sex == 'male') & (combine_df.Age >= 12)
#                                              & (combine_df.Survived == 1) & (combine_df.Familysize > 1)][
#                                       'Fname'].values))
#     combine_df['Dead_female_family'] = np.where(combine_df['Fname'].isin(dead_female_Fname), 1, 0)
#     combine_df['Survive_male_family'] = np.where(combine_df['Fname'].isin(survive_male_Fname), 1, 0)
#     combine_df = combine_df.drop(['Name', 'Fname'], axis=1)
#
#
#     #Age  添加一个小孩子标签
#     # print(combine_df.Age)
#     group = combine_df.groupby(['Title', 'Pclass'])['Age']
#     # print(group.count())
#     print('******************************************')
#     combine_df['Age'] = group.transform(lambda x: x.fillna(x.median()))
#
#     combine_df = combine_df.drop('Title', axis=1)
#
#     combine_df['IsChild'] = np.where(combine_df['Age'] <= 12, 1, 0)
#     # print(combine_df.columns)
#     combine_df['Age'] = pd.cut(combine_df['Age'], 5)
#     combine_df = combine_df.drop('Age', axis=1)
#
#     #Familysize    我们将上面提取过的Familysize再离散化
#     combine_df['Familysize'] = np.where(combine_df['Familysize'] == 0, 'solo',
#                                         np.where(combine_df['Familysize'] <= 3, 'normal', 'big'))
#     df = pd.get_dummies(combine_df['Familysize'], prefix='Familysize')
#     combine_df = pd.concat([combine_df, df], axis=1).drop(['SibSp', 'Parch', 'Familysize'],axis = 1)
#     # print(combine_df.columns)
#     # print(len(combine_df.columns))
#
#
#     #Ticket
#     combine_df['Ticket_Lett'] = combine_df['Ticket'].apply(lambda x: str(x)[0])
#     combine_df['Ticket_Lett'] = combine_df['Ticket_Lett'].apply(lambda x: str(x))
#
#     combine_df['High_Survival_Ticket'] = np.where(combine_df['Ticket_Lett'].isin(['1', '2', 'P']), 1, 0)
#     combine_df['Low_Survival_Ticket'] = np.where(combine_df['Ticket_Lett'].isin(['A', 'W', '3', '7']), 1, 0)
#     combine_df = combine_df.drop(['Ticket', 'Ticket_Lett'], axis=1)
#
#     #Embarked
#     combine_df.Embarked = combine_df.Embarked.fillna('S')
#     df = pd.get_dummies(combine_df['Embarked'], prefix='Embarked')
#     combine_df = pd.concat([combine_df, df], axis=1).drop('Embarked', axis=1)
#
#     #Cabin
#     combine_df['Cabin_isNull'] = np.where(combine_df['Cabin'].isnull(), 0, 1)
#     combine_df = combine_df.drop('Cabin', axis=1)
#
#
#     #Pclass
#     df = pd.get_dummies(combine_df['Pclass'],prefix = 'Pclass')
#     combine_df = pd.concat([combine_df,df],axis = 1).drop('Pclass',axis = 1)
#
#     #Sex
#     df = pd.get_dummies(combine_df['Sex'], prefix='Sex')
#     combine_df = pd.concat([combine_df, df], axis=1).drop('Sex', axis=1)
#
#     #
#     # #Fare  缺省值用众数填充，之后进行离散化
#     # combine_df['Fare'] = pd.qcut(combine_df.Fare, 3)
#     # # print(combine_df.Fare)
#     # df = pd.get_dummies(combine_df.Fare, prefix='Fare').drop('Fare_(-0.001, 8.662]', axis=1)
#     # combine_df = pd.concat([combine_df, df], axis=1).drop('Fare', axis=1)
#
#
#     # Fare
#     combine_df['Fare'].fillna(combine_df['Fare'].dropna().median(), inplace=True)
#     combine_df['Low_Fare'] = np.where(combine_df['Fare'] <= 8.662, 1, 0)
#     combine_df['High_Fare'] = np.where(combine_df['Fare'] >= 26, 1, 0)
#     combine_df = combine_df.drop('Fare', axis=1)
#
#
#
#     print(combine_df.columns.values)
#     print(len(combine_df.columns.values))
#
#
#     print('before LabelEncoder.....')
#     print(combine_df)
#     print(len(combine_df.columns))
#
#     #所有特征转化成数值型编码
#     features = combine_df.drop(["PassengerId", "Survived"], axis=1).columns
#     le = LabelEncoder()
#     for feature in features:
#         le = le.fit(combine_df[feature])
#         combine_df[feature] = le.transform(combine_df[feature])
#
#     print('After LabelEncoder....')
#     print(combine_df)
#
#     # 防止xgboost 不识别特征中的  ']'
#     combine_df.rename(columns=lambda x: x.replace('[', '').replace(']',''), inplace=True)
#
#     #得到训练/测试数据
#     X_all = combine_df.iloc[:891, :].drop(["PassengerId", "Survived"], axis=1)
#     Y_all = combine_df.iloc[:891, :]["Survived"]
#     X_test = combine_df.iloc[891:, :].drop(["PassengerId", "Survived"], axis=1)
#     # print('X_all = ')
#     # print(X_all)
#     # print(Y_all)
#     # print(X_test)
#     # print(X_all.columns.values)
#     # print(X_test.columns.values)
#     print('*******************************************************')
#
#
#     #
#     #分别考察逻辑回归、支持向量机、最近邻、决策树、随机森林、gbdt、xgbGBDT几类算法的性能。
#     param_lr = {'penalty': ['l1', 'l2'], 'C': [0.1, 0.5, 5.0]}
#     grd_lr = GridSearchCV(estimator = LogisticRegression(),param_grid = param_lr,cv = 5,n_jobs = -1)
#     grd_lr.fit(X_all,Y_all)
#     # print(grd_lr.best_estimator_,grd_lr.best_score_)
#     lr = grd_lr.best_estimator_
#
#     param_svc = {'C':[0.1,1.0,10.0],'gamma':['auto',1,0.1,0.01],'kernel':['linear','rbf','sigmoid']}
#     grd_svc = GridSearchCV(estimator = SVC(),param_grid = param_svc,cv = 5,n_jobs = -1)
#     grd_svc.fit(X_all,Y_all)
#     # print(grd_svc.best_estimator_,grd_svc.best_score_)
#     svc = grd_svc.best_estimator_
#
#     k_range = list(range(1, 10))
#     leaf_range = list(range(1, 2))
#     weight_options = ['uniform', 'distance']
#     algorithm_options = ['auto', 'ball_tree', 'kd_tree', 'brute']
#     param_knn = dict(n_neighbors=k_range, weights=weight_options, algorithm=algorithm_options)
#     grd_knn = GridSearchCV(estimator = KNeighborsClassifier(),param_grid = param_knn,cv = 5,n_jobs = -1)
#     grd_knn.fit(X_all,Y_all)
#     # print(grd_knn.best_estimator_,grd_knn.best_score_)
#     knn = grd_knn.best_estimator_
#
#     param_dt = {'max_depth': [1, 2, 3, 4, 5], 'max_features': [1, 2, 3, 4]}
#     grd_dt = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=param_dt, cv=5, n_jobs=-1)
#     grd_dt.fit(X_all, Y_all)
#     # print(grd_dt.best_estimator_, grd_dt.best_score_)
#     dt = grd_dt.best_estimator_
#
#     param_rf = {'n_estimators': [50,100,300,500], 'min_samples_leaf': [1, 2, 3, 4],'class_weight':[{0: 0.745, 1: 0.255}]}
#     grd_rf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_rf, cv=5, n_jobs=-1)
#     grd_rf.fit(X_all, Y_all)
#     # print(grd_rf.best_estimator_, grd_rf.best_score_)
#     rf = grd_rf.best_estimator_
#
#     gbdt = GradientBoostingClassifier(n_estimators=500, learning_rate=0.03, max_depth=3)
#     xgb = XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
#     # clfs = [('lr', lr), ('svc', svc), ('knn', knn), ('dt', dt), ('rf', rf), ('gbdt', gbdt),('xgb',xgb)]
#     #
#     # kfold = 10
#     # cv_results = []
#     # for classifier in clfs:
#     #     cv_results.append(cross_val_score(classifier, X_all, y=Y_all, scoring="accuracy", cv=kfold, n_jobs=4))
#
#     # for classifier in clfs:
#     #     print('the cross_value of {} is:'.format(classifier[0]))
#     #     # print(cross_val_score(classifier[1], X_all, y=Y_all, scoring="accuracy", cv=kfold, n_jobs=4))
#     #     cross_result = list(cross_validation.cross_val_score(classifier[1], X_all, Y_all, cv=5))
#     #     cross_result.append(np.mean(cross_result))
#     #     print(cross_result)
#
#
#
#
#
#     #
#     # #XGBClassifier 很坑，属性名中不能有 '[,]'
#     # demo_train = X_all.iloc[:,:]           #是深拷贝，怎么实现浅拷贝？
#     # demo_train['Fare_1'] = X_all['Fare_(8.662, 26.0]']
#     # demo_train['Fare_2'] = X_all['Fare_(26.0, 512.329]']
#     # demo_test = X_test.iloc[:,:]
#     # demo_test['Fare_1'] = X_all['Fare_(8.662, 26.0]']
#     # demo_test['Fare_2'] = X_all['Fare_(26.0, 512.329]']
#     # demo_test = demo_test.drop(['Fare_(8.662, 26.0]', 'Fare_(26.0, 512.329]'], axis=1)
#     # print(X_all.columns.values)
#     # print(demo_train.columns.values)
#     # demo_train = demo_train.drop(['Fare_(8.662, 26.0]','Fare_(26.0, 512.329]'],axis = 1)
#     # print(demo_train.columns.values)
#     # print(cross_val_score(rf, demo_train, y=Y_all, scoring="accuracy", cv=kfold, n_jobs=4))
#     # rf.fit(demo_train,Y_all)
#     # print(rf.feature_importances_)
#     # print(type(rf.feature_importances_))
#     # print(rf.feature_importances_.sum())
#
#     #
#     # #实现浅拷贝X_all
#     # demo = X_all.drop(['Fare_(8.662, 26.0]','Fare_(26.0, 512.329]'],axis = 1)
#     # print(demo.columns.values)
#     # print(X_all.columns.values)
#     # demo['Fare1'] = X_all['Fare_(8.662, 26.0]']
#     # demo['Fare2'] = X_all['Fare_(26.0, 512.329]']
#     # print(demo.columns.values)
#     # print(X_all.columns.values)
#
#     # cv_means = []
#     # cv_std = []
#     # for cv_result in cv_results:
#     #     cv_means.append(cv_result.mean())
#     #     cv_std.append(cv_result.std())
#
#     # cv_res = pd.DataFrame({"CrossValMeans": cv_means, "CrossValerrors": cv_std,
#     #                        "Algorithm": ["LR", "SVC", 'KNN', 'decision_tree', "random_forest", "GBDT", "xgbGBDT"]})
#
#     # # g = sns.barplot("CrossValMeans", "Algorithm", data=cv_res, palette="Set3", orient="h", **{'xerr': cv_std})
#     # # g.set_xlabel("Mean Accuracy")
#     # # g = g.set_title("Cross validation scores")
#
#
#
#
#
#
#     #观察发现不同的模型的feature importance 有比较大的差别，，，把他们组合再一起会不会更好呢
#
#
#
#     #集成框架Ensemble，我们把基分类器丢进去。
#     # bag = Ensemble([('xgb', xgb), ('lr', lr), ('rf', rf), ('svc', svc), ('gbdt', gbdt)])
#     # bag = Ensemble([('xgb', xgb), ('lr', lr), ('gbdt', gbdt), ('rf', rf)])
#     # score = 0
#     # for i in range(0, 10):
#     #     num_test = 0.20
#     #     X_train, X_cv, Y_train, Y_cv = train_test_split(X_all, Y_all, test_size=num_test)
#     #     bag.fit(X_train, Y_train)
#     #     # Y_test = bag.predict(X_test)
#     #     acc_xgb = round(bag.score(X_cv, Y_cv) * 100, 2)
#     #     score += acc_xgb
#     # print(score / 10)  # 0.8786
#     #
#     # print(X_all.values)
#
#     x = X_all.as_matrix()
#     y = Y_all.as_matrix()
#     test = X_test.as_matrix()
#     # stackings = [('lr', lr), ('svc', svc), ('knn', knn), ('dt', dt), ('rf', rf), ('gbdt', gbdt),('xgb',xgb)]
#     first_estimators =[('lr', lr), ('svc', svc), ('knn', knn), ('dt', dt), ('rf', rf), ('gbdt', gbdt),('xgb',xgb)]
#
#     # param = {'penalty':['l1','l2'],'C':[0.1,0.5,5.0]}
#     # #use best estimator
#     # my_lr = lr2.best_estimator_
#     # my_lr.fit(middle_train,y)
#     # predictions = my_lr.predict(middle_test)
#     sta = Stacking(first_estimators)
#     print(sta.score(x,y))
#     predictions = sta.fit_predict(x,y,test,Kfold = 10)
#
#     # lr2.fit(middle_train,y)
#     # predictions = lr2.predict(middle_test)
#     # print(predictsions)
#     # print(predictsions.shape)
#     # print(cross_validation.cross_val_score(lr2, middle_train, y, cv=5))
#
#     #stacking saving
#     # result = pd.DataFrame(
#     #     {'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
#     # result.to_csv('D:/titanic/results/' +'stacking'+'_predictions' + time.strftime('%Y-%m-%d %H-%M-%S',
#     #                                                                            time.localtime(time.time())) + '.csv',
#     #               index=False)
#
#
#
#
#
#
#
#     #
#     # bag.fit(X_all,Y_all)
#     # predictions = bag.predict(X_test)
#     # result = pd.DataFrame(
#     #     {'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
#     # result.to_csv('D:/titanic/results/' +'bag' +'_predictions' + time.strftime('%Y-%m-%d %H-%M-%S',
#     #                                                                            time.localtime(time.time())) + '.csv',
#     #               index=False)
#     #
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#     # 预测并打印结果到csv
#     # for i in clfs:
#         # i[1].fit(X_all,Y_all)
#         # predictions = i[1].predict(X_test)
#         # result = pd.DataFrame(
#         #     {'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
#         # result.to_csv('D:/titanic/results/' +i[0] +'_predictions' + time.strftime('%Y-%m-%d %H-%M-%S',
#         #                                                                            time.localtime(time.time())) + '.csv',
#         #               index=False)
#
#
#
#
#
#
#
#
#
#
# #
# # class A(object):
# #     def __init__(self,c = 1):
# #         self.b = 0
# #     def cccccc(self):
# #         self.cccccc = 10
# #     def print_(self):
# #         print('b = {}'.format(self.b))
# #         print('ccccccc = {}'.format(self.cccccc))
# #         print('c = {}'.format(c))
# #
# #
# # aaa = A(100)
# # aaa.print_()


class BinaryTreeNode(object):
    def __init__(self,data,lchild = None,rchild = None):
        self.data = data
        self.lchild = lchild
        self.rchild = rchild


class BinaryTree(object):
    def __init__(self,root = None):
        self.root = root
        self.pre,self.in0,self.post= [],[],[]

    def isEmpty(self):
        return self.root==None

    def preOrder(self,node):
        if node ==self.root:
            self.pre = []
        if node:
            self.pre.append(node.data)
            self.preOrder(node.lchild)
            self.preOrder(node.rchild)
        return self.pre

    def inOrder(self,node):
        if node == self.root:
            self.in0 = []
        if node:
            self.inOrder(node.lchild)
            self.in0.append(node.data)
            self.inOrder(node.rchild)
        return self.in0

    def postOrder(self,node):
        if node == self.root:
            self.post = []
        if node:
            self.postOrder(node.lchild)
            self.postOrder(node.rchild)
            self.post.append(node.data)
        return self.post



def create_BinaryTree_by_pre_and_in0(pre,in0):
    if len(pre)==0:
        return None
    if len(pre)==1:
        return BinaryTreeNode(pre[0])
    root = BinaryTreeNode(pre[0])
    loc = in0.index(root.data)
    root.lchild = create_BinaryTree_by_pre_and_in0(pre[1:loc+1],in0[:loc])
    root.rchild = create_BinaryTree_by_pre_and_in0(pre[loc+1:],in0[loc+1:])
    return root



# pre = [1,2,4,7,3,5,6,8]
# in0 = [4,7,2,1,5,3,8,6]
# bt_root = create_BinaryTree_by_pre_and_in0(pre,in0)
# bt = BinaryTree(bt_root)
# print(bt.preOrder(bt.root))
# print(bt.inOrder(bt.root))
# print(bt.postOrder(bt.root))

# #图的遍历
#
# G={'A':{'A':0,'B':1,'C':1,'D':1},
#    'B':{'B':0,'E':1},
#    'C':{'C':0,'D':1,'B':1},
#    'D':{'A':1,'C':1,'D':0},
#    'E':{'E':0}}
# path = dict((key,False) for key in G)
# print(path)
# def DFS(v):
#     print(v)
#     path[v] = True
#     for i in G[v]:
#         if not path[i] and G[v][i]:
#             DFS(i)
#
#
# def BFS(v):
#     path = dict((key,False) for key in G)
#     queue = [v]
#     while queue:
#         k = queue.pop(0)
#         if not path[k]:
#             print(k)
#             path[k] =True
#         for i in G[k]:
#             if not path[i] and G[k][i]:
#                 queue.append(i)
# # DFS('A')
# BFS('A')






# def judge_prime(x):
#     if x<=1:
#         return False
#     for i in range(2,int(x**0.5+1)):
#         if x%i==0:
#             return False
#
#     return True
#
#
# if __name__ =='__main__':
#     q = int(raw_input())
#     max = 0
#     result,li = [],[]
#     for i in range(q):
#         a = int(raw_input())
#         result.append(a)
#         if max<a:
#             max = a
#     i = 2
#     while len(li)<max:
#         if judge_prime(i):
#             li.append(i)
#         i+=1
#     # print(li)
#     for x in range(q):
#         print(li[result[x]-1])


# if __name__ == '__main__':
#     W = []
#     stack = []
#     sum = 0
#     for i in range(len(W)):
#         sum += W[i]
#     result = sum
#     results = []
#     T = 0  # T表示当前组糖的总重量
#     k = 0  # k表示当前搜索到第几个数
#     line = raw_input().strip().split()
#     for x in range(1,len(line)):
#         W.append(int(line[x]))
#         sum +=int(line[x])
#     # print(W)
#     # print(sum)
#     diff = sum  # diff表示两组糖重量差距
#     while stack or k < len(W):
#         if len(stack) == 0:
#             stack.append(k)
#             T += W[k]
#             k += 1
#
#         while k < len(W):
#             if len(stack) + 1<= len(W) // 2 + len(W)%2:  # 只要能使两组糖数相差不大于1就作为待选分组情况
#                 stack.append(k)
#                 T += W[k]
#             k += 1
#
#             if abs(sum - 2 * T) < diff :  # 如果该分组能减少两组差值diff就更新diff及result
#                 result = T
#                 diff = abs(sum - 2 * T)
#                 results = list(stack)
#
#         k = stack.pop()  # 栈顶元素出栈，回溯，继续试探下一个数
#         T -= W[k]
#         k += 1
#
#     print min(result, sum - result), max(result, sum - result)    #输出两组糖的重量，小的一组在前
#     # print(results)





def select_sweets(T,pos):
    if pos == len(W):
        return
    select_sweets(T,pos + 1)
    if len(stack) + 1<= len(W)//2 + len(W)%2:
        stack.append(pos)
        T += W[pos]
        if abs(sum - 2*T) < result[2]:  #若符合要求即改变result，diff
            result[0] = T
            result[1] = sum - T
            result[2] = abs(sum - 2*T)
        select_sweets(T,pos+1)
        stack.pop()

if __name__ == '__main__':
    W,stack = [],[]
    sum = 0
    T = 0  # T表示当前组糖的总重量
    k = 0  # k表示当前搜索到第几个数
    line = raw_input().strip().split()
    for x in range(1, len(line)):
        W.append(int(line[x]))
        sum += int(line[x])
    diff = sum
    result = [0, sum, diff]     #要注意函数内部不可改变全局变量的值，但列表全局可用，所以把它放在全局列表当中就解决了
    # print(result)
    select_sweets(T, 0)
    print min(result[0], result[1]),max(result[0], result[1])














































