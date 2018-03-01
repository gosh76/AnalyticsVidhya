import numpy as np
import pandas as pd
import csv
from sklearn.ensemble import GradientBoostingClassifier
trn = pd.read_csv('~/Mckinsey1/train.csv')
trn.head()
trn.isnull().sum()
tst = pd.read_csv('~/Mckinsey1/test.csv')
tst.head()
tst.isnull().sum()
target = trn.Approved
ids = tst.ID
trn = trn.drop(['Approved'],axis=1)
c1 = pd.concat([trn,tst])
c1.loc[c1.DOB.isnull(),'DOB'] = '23/07/79'
c1.loc[c1.City_Code.isnull(),'City_Code'] = c1.City_Code.mode()[0]
c1.loc[c1.City_Category.isnull(),'City_Category'] = c1.City_Category.mode()[0]
c1.loc[c1.Employer_Code.isnull(),'Employer_Code'] = c1.Employer_Code.mode()[0]
c1.loc[c1.Employer_Category1.isnull(),'Employer_Category1'] = c1.Employer_Category1.mode()[0]
c1.loc[c1.Employer_Category2.isnull(),'Employer_Category2'] = c1.Employer_Category2.median()
c1.loc[c1.Customer_Existing_Primary_Bank_Code.isnull(),'Customer_Existing_Primary_Bank_Code'] = c1.Customer_Existing_Primary_Bank_Code.mode()[0]
c1.loc[c1.Primary_Bank_Type.isnull(),'Primary_Bank_Type'] = c1.Primary_Bank_Type.mode()[0]
c1.loc[c1.Existing_EMI.isnull(),'Existing_EMI'] = c1.Existing_EMI.median()
#c1.loc[c1.Loan_Amount.isnull(),'Loan_Amount'] = c1.Loan_Amount.median()
c1.loc[c1.Loan_Amount.isnull() & c1.Monthly_Income<=1300,'Loan_Amount'] = 20000
c1.loc[c1.Loan_Amount.isnull() & c1.Monthly_Income<=1500,'Loan_Amount'] = 30000
c1.loc[c1.Loan_Amount.isnull() & c1.Monthly_Income<=2000,'Loan_Amount'] = 40000
c1.loc[c1.Loan_Amount.isnull() & c1.Monthly_Income<=2500,'Loan_Amount'] = 50000
c1.loc[c1.Loan_Amount.isnull() & c1.Monthly_Income<=3000,'Loan_Amount'] = 80000
c1.loc[c1.Loan_Amount.isnull() & c1.Monthly_Income<=3500,'Loan_Amount'] = 90000
c1.loc[c1.Loan_Amount.isnull() & c1.Monthly_Income<=4000,'Loan_Amount'] = 100000
c1.loc[c1.Loan_Amount.isnull() & c1.Monthly_Income<=5000,'Loan_Amount'] = 150000
c1.loc[c1.Loan_Amount.isnull() & c1.Monthly_Income<=7500,'Loan_Amount'] = 180000
c1.loc[c1.Loan_Amount.isnull() & c1.Monthly_Income>7500,'Loan_Amount'] = 200000
#c1.loc[c1.Loan_Period.isnull(),'Loan_Period'] = c1.Loan_Period.median()
c1.loc[c1.Loan_Period.isnull() & c1.Loan_Amount<=80000,'Loan_Period'] = 1
c1.loc[c1.Loan_Period.isnull() & c1.Loan_Amount<=120000,'Loan_Period'] = 2
c1.loc[c1.Loan_Period.isnull() & c1.Loan_Amount<=180000,'Loan_Period'] = 3
c1.loc[c1.Loan_Period.isnull() & c1.Loan_Amount<=200000,'Loan_Period'] = 4
c1.loc[c1.Loan_Period.isnull() & c1.Loan_Amount>200000,'Loan_Period'] = 5
c1.loc[c1.Interest_Rate.isnull(),'Interest_Rate'] = c1.Interest_Rate.median()
c1.loc[c1.EMI.isnull(),'EMI'] = c1.EMI.median()
c1['d1'] = [int(str(c1.DOB.values[i])[:2]) for i in range(len(c1))]
c1['m1'] = [int(str(c1.DOB.values[i])[3:5]) for i in range(len(c1))]
c1['y1'] = [int(str(c1.DOB.values[i])[6:]) for i in range(len(c1))]
c1 = c1.drop(['DOB'],axis=1)
c1['d2'] = [int(str(c1.Lead_Creation_Date.values[i])[:2]) for i in range(len(c1))]
c1['m2'] = [int(str(c1.Lead_Creation_Date.values[i])[3:5]) for i in range(len(c1))]
c1['y2'] = [int(str(c1.Lead_Creation_Date.values[i])[6:]) for i in range(len(c1))]
c1 = c1.drop(['Lead_Creation_Date'],axis=1)
c1['Gender'] = c1.Gender.astype('category').cat.codes
c1['City_Code'] = c1.City_Code.astype('category').cat.codes
c1['City_Category'] = c1.City_Category.astype('category').cat.codes
c1['Employer_Code'] = c1.Employer_Code.astype('category').cat.codes
c1['Employer_Category1'] = c1.Employer_Category1.astype('category').cat.codes
c1['Customer_Existing_Primary_Bank_Code'] = c1.Customer_Existing_Primary_Bank_Code.astype('category').cat.codes
c1['Primary_Bank_Type'] = c1.Primary_Bank_Type.astype('category').cat.codes
c1['Contacted'] = c1.Contacted.astype('category').cat.codes
c1['Source'] = c1.Source.astype('category').cat.codes
c1['Source_Category'] = c1.Source_Category.astype('category').cat.codes
c1 = c1.drop(['ID'],axis=1)
train = c1.head(69713)
test = c1.tail(30037)
del trn,tst,c1
tra = np.array(train,order='C',copy=False)
tar = np.array(target,order='C',copy=False)
tes = np.array(test,order='C',copy=False)

model = GradientBoostingClassifier(loss='deviance',learning_rate=0.1,n_estimators=150,subsample=1.0,criterion='friedman_mse',min_samples_split=3,min_samples_leaf=1,\
min_weight_fraction_leaf=0.0,max_depth=5,min_impurity_split=None,init=None,random_state=79,max_features=None,verbose=2,max_leaf_nodes=None,warm_start=False,\
presort='auto')
model = model.fit(tra,tar)
pred = model.predict(tes)

sub1 = pd.DataFrame(ids,columns=['ID'])
sub1['Approved'] = pred
sub1.to_csv('Mckinseyh7.csv',index=False)

