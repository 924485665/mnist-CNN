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
import warnings
warnings.filterwarnings('ignore')





data_train = pd.read_csv("D:/titanic/train.csv")
# print(data_train.columns)
# print('\n')
# print(data_train.info())
# print('\n')
# print(data_train.describe())
# print('\n')
# print(data_train[data_train.Cabin.notnull()]['Survived'].value_counts())
# print('\n')
# print(data_train[data_train.Cabin.notnull()]['Survived'])
# 看看各性别的获救情况
# fig = plt.figure()
# fig.set(alpha= 0.2)
#
#
# plt.subplot2grid((2,2),(0,0))
# data_train.Survived.value_counts().plot(kind = 'bar')
# plt.title(u'获救情况')
# plt.ylabel(u'人数')
#
# plt.subplot2grid((2,2),(0,1))
# data_train.Pclass.value_counts().plot(kind = 'bar')
# plt.title(u'乘客等级分布')
# plt.ylabel(u'人数')
#
#
# plt.subplot2grid((2,2),(1,0),colspan=2)
# data_train.Age[data_train.Pclass == 3].plot(kind = 'kde')
# data_train.Age[data_train.Pclass == 2].plot(kind = 'kde')
# data_train.Age[data_train.Pclass == 1].plot(kind = 'kde')
# plt.xlabel(u'年龄')
# plt.ylabel(u'密度')
# plt.title(u'各阶层乘客的年龄分布')
# plt.legend((u'3阶层',u'2阶层',u'1阶层'),loc = 'best')
# plt.savefig("D:/titanic/feature_pics/"+ time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time())) +".png")
# plt.show()

# survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
# survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
# print(survived_m)
# df_sex = pd.DataFrame({u'男性':survived_m,u'女性':survived_f})
# df_sex.plot(kind = 'bar',stacked = True)
# plt.title(u'由性别看获救情况')
# plt.ylabel(u'人数')



# plt.subplot2grid((2,4),(1,0),colspan=2)
# survived_00 = data_train.Pclass[data_train.Survived == 0].value_counts()
# # print(survived_0)
# # print(type(survived_0))
# survived_11 = data_train.Pclass[data_train.Survived == 1].value_counts()
# df = pd.DataFrame({u'获救':survived_11,u'未获救':survived_00})
# # print(df)
# # print(type(df))
# df.plot(kind = 'bar',stacked = True)
# plt.ylabel(u'人数')
# plt.title(u'乘客等级及存活分布')
#
# plt.subplot2grid((2,4),(1,2),colspan=2)
# survived_000 = data_train.Embarked[data_train.Survived == 0].value_counts()
# survived_111 = data_train.Embarked[data_train.Survived == 1].value_counts()
# df_Embarked = pd.DataFrame({u'获救':survived_111,u'未获救':survived_000})
# df_Embarked.plot(kind = 'bar',stacked = True)
# plt.title(u'由登岸港口看获救情况')
# plt.ylabel(u'人数')

# plt.subplot2grid((2,6),(1,2),colspan=2)
# data_train.Age[data_train.Embarked == 'S'].plot(kind = 'kde')
# data_train.Age[data_train.Embarked == 'C'].plot(kind = 'kde')
# data_train.Age[data_train.Embarked == 'Q'].plot(kind = 'kde')
# plt.xlabel(u'年龄')
# plt.ylabel(u'密度')
# plt.title(u'各登岸港口乘客的年龄分布')
# plt.legend((u'S港口',u'C港口',u'Q港口'),loc = 'best')

# plt.subplot2grid((2,4),(1,4),colspan=2)
# data_train.Age[data_train.Pclass == 3].plot(kind = 'kde')
# data_train.Age[data_train.Pclass == 2].plot(kind = 'kde')
# data_train.Age[data_train.Pclass == 1].plot(kind = 'kde')
# plt.xlabel(u'年龄')
# plt.ylabel(u'密度')
# plt.title(u'各阶层乘客的年龄分布')
# plt.legend((u'3阶层',u'2阶层',u'1阶层'),loc = 'best')

# plt.scatter(data_train.Survived,data_train.Age,c = 'r')
# plt.ylabel(u'年龄')
# plt.grid(b=True,which ='major',axis= 'y')
# plt.title(u'按年龄看获救分布 (1为获救)')
# plt.show()
from numpy.random import randn
# df = DataFrame(randn(5,2),index=range(0,10,2),columns=list('AB'))
# print(df)
# print(df.iloc[[2]])
# print(df.loc[[2]])
svc =1
print(str(svc))