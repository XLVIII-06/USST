import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


#获取数据
file_dir = 'C:/Users/12646/Desktop/Graduation Project/Data/3ms/csv/unlabeled/'  # file directory
all_csv_list = os.listdir(file_dir)  # get csv list
for single_csv in all_csv_list:
    single_data_frame = pd.read_csv(os.path.join(file_dir, single_csv))
#     print(single_data_frame.info())
    if single_csv == all_csv_list[0]:
        data = single_data_frame
    else:  # concatenate all csv to a single dataframe, ingore index
        data = pd.concat([data, single_data_frame], ignore_index=True)


#获取数据
#data = pd.read_csv('C:/Users/12646/Desktop/Graduation Project/Data/3ms/csv/unlabeled/WSN_20211229_164114(无干扰3ms).csv')


#数据预处理，对缺失值进行合理填充
data['rssi'] = data['rssi'].fillna(value = data['rssi'].min())
data['lqi'] = data['lqi'].fillna(value = data['lqi'].min())
data['snr'] = data['snr'].fillna(value = data['snr'].min())


#取滑动平均值，滑动窗口为5
data['rssirol'] = data['rssi'].rolling(window = 5, min_periods=1).mean()
data['lqirol'] = data['lqi'].rolling(window = 5, min_periods=1).mean()
data['snrrol'] = data['snr'].rolling(window = 5, min_periods=1).mean()


#划分训练集与测试集（7：3）
x = data[['rssi','lqirol','snr']]
y = data['Prr']
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3)


#训练决策树回归模型
treereg = DecisionTreeRegressor()
treereg.fit(train_x,train_y)


#预测
pred = treereg.predict(test_x)


#对模型进行评估R2，MSE
r2 = r2_score(test_y,pred)
mse = mean_squared_error(test_y,pred)

print('R2 = ',r2)
print('MSE = ',mse)