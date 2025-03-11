import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


data = pd.read_csv('C:/Users/12646/Desktop/Graduation Project/Data/3ms/csv/labeled/WSN_20211229_164114(无干扰3ms).csv')
data['rssi'] = data['rssi'].fillna(data['rssi'].min())


#构建决策树
#features=data.iloc[:,:-2]
features=data['rssi']
labels=data['label']
#print(labels)
print(data['label'].value_counts() )
train_x,test_x,train_y,test_y=train_test_split(features,labels,test_size=0.3)
#生成决策树
model=DecisionTreeClassifier(criterion='gini',max_depth=10,min_samples_leaf=10,min_samples_split=20,random_state=9)
#训练模型
model.fit(train_x.values.reshape(-1,1),train_y)
#预测
pre_y=model.predict(test_x.values.reshape(-1,1))
print("pre_y:",pre_y)
print("test_y:",test_y)
#模型得分
print('模型得分：',accuracy_score(test_y,pre_y))