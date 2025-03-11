import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



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


#选择需要绘制散点图的列名 
cols = ["rssirol","lqirol","snrrol","Prr"]
#通过seaborn绘制散点图 
sns.set(style="whitegrid",context="notebook")
sns.pairplot(data[cols],height=2) 
plt.show()


#获取相关系数矩阵
cc=data[cols].corr()
plt.subplots(figsize=(6, 6))
#绘制相关系数图
sns.heatmap(cc, annot=True,  square=True, cmap="Blues")
plt.show()