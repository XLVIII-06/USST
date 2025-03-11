import pandas as pd

cols=["rssi","snr","lqi","Prr"]
def readData(flist):
    for i in range(0,len(flist),1):
        data = pd.read_csv(flist[i],engine='python')
        if i==0:
            dataset=data[cols] 
        else:
            dataset=pd.concat([dataset,data[cols]],ignore_index=True) 
            return dataset

