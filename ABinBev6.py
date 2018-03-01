import numpy as np
import pandas as pd
import csv
#train
hv = pd.read_csv('F:/ABinBev/train/historical_volume.csv')
hv['y'] = [str(hv.YearMonth.values[i])[:4] for i in range(len(hv))]
hv['m'] = [str(hv.YearMonth.values[i])[4:] for i in range(len(hv))]
hv.Agency = hv.Agency.astype('category').cat.codes
hv.SKU = hv.SKU.astype('category').cat.codes
target = hv.Volume
hv = hv.drop(['Volume','YearMonth'],axis=1)
#test
tst = pd.read_csv('F:/ABinBev/test/volume_forecast.csv')
tst['YearMonth'] = 201801
tst['y'] = [str(tst.YearMonth.values[i])[:4] for i in range(len(tst))]
tst['m'] = [str(tst.YearMonth.values[i])[4:] for i in range(len(tst))]
tst = tst.drop(['YearMonth','Volume'],axis=1)
agencyid = tst.Agency
skuid = tst.SKU
tst.Agency = tst.Agency.astype('category').cat.codes
tst.SKU = tst.SKU.astype('category').cat.codes
#model
tra = np.array(hv,order='C',copy=False)
tar = np.array(target,order='C',copy=False)
tes = np.array(tst,order='C',copy=False)
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(loss='lad',n_estimators=150,random_state=1029,verbose=2,max_depth=7)
model = model.fit(tra,tar)
pred = model.predict(tes)
sub1 = pd.DataFrame(agencyid,columns=['Agency'])
sub1['SKU'] = skuid
sub1['Volume'] = pred
sub1.loc[sub1.Volume<0,'Volume'] = 0
sub1.to_csv('F:/ABinBev/volume_forecast.csv',index=False)
#SKUs
hv1 = pd.read_csv('F:/ABinBev/train/historical_volume.csv')
psp = pd.read_csv('F:/ABinBev/train/price_sales_promotion.csv')
t1 = pd.merge(hv1,psp,on=['Agency','SKU','YearMonth'],how='inner')
t1['Revenue'] = t1['Volume']*t1['Sales']
list1 = [201701,201702,201703,201704,201705,201706,201707,201708,201709,201710,201711,201712]
t2 = t1.loc[t1.YearMonth.isin(list1),:]
tf1 = t2.groupby('SKU')['Revenue'].agg(np.mean)
tf2 = tf1.sort_values(ascending=False)
tf3 = list(tf2.index)

