import pandas as pd
from sklearn.preprocessing import scale,Normalizer
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import warnings
warnings.simplefilter(action='ignore', category=any)
import infinity
import matplotlib.pyplot as plt

# read .csv file
train_data = pd.read_csv("S2 File.csv", sep=',', index_col=False,usecols=['Sex','Age','DIC','APACHE2','Anticoagulation','Thrombosis','Bleeding'
    ,'organ failure','SIRS','Infection','Tissue','Surgery','Hema','Cancer','Liver','Obstetric','Vascular','Immunologic','organ transplant','ISTH','JMHW','JAAM','PT','PT(%)','INR','aPTT','Fibrinogen','D-Dimer','TT','AT III','FDP','WBC','RBC','Hb','Hct','MCV','MCH','MCHC','RDW','PDW (%)','PDW (fL)','MPV','DNI','TMS','PLT','Neu(%)','Lympho(%)','Mono(%)','Eo(%)','Baso(%)','LUC(%)'])
test_data2 = pd.read_csv("set2_DIC.csv", sep=',', index_col=False,usecols=['Sex','Age','APACHE2','Anticoagulation','Thrombosis','Bleeding'
    ,'organ failure','SIRS','Infection','Tissue','Surgery','Hema','Cancer','Liver','Obstetric','Vascular','Immunologic','organ transplant','ISTH','JMHW','JAAM','PT','PT(%)','INR','aPTT','Fibrinogen','D-Dimer','TT','AT III','FDP','WBC','RBC','Hb','Hct','MCV','MCH','MCHC','RDW','PDW (%)','PDW (fL)','MPV','DNI','TMS','PLT','Neu(%)','Lympho(%)','Mono(%)','Eo(%)','Baso(%)','LUC(%)'])
#df_combined = pd.concat([data,data2])
train_data.replace('-', np.nan, inplace=True)
train_data.replace(infinity, np.nan, inplace=True)
train_data.replace('M', 2, inplace=True)
train_data.replace('F',1, inplace=True)
cols = train_data.columns[train_data.dtypes.eq('object')]
train_data[cols] = train_data[cols].apply(pd.to_numeric, errors='coerce')
train_data.reset_index()
print(train_data .head(1500))
print(train_data .dtypes)

# preprocessing
X = train_data.drop(columns=['DIC'])
x_arr = np.asarray(X)
#print(x_arr.shape)

imputer = KNNImputer(n_neighbors=11, weights="uniform")
x_arr=imputer.fit_transform(x_arr)
#print(x_arr)

np.savetxt('After_preprocessing.csv',x_arr, delimiter=',', fmt='%s')

X_scaled = scale(x_arr)
#print(X_scaled)

y = train_data[['DIC']]
y_arr = np.asarray(y)
y_arr=np.ravel(y_arr, order='C')
print(y_arr.shape)

clf =KNeighborsClassifier()
clf.fit(X_scaled, y_arr)
clf.predict(X_scaled)
print("The accuracy of KNN Classifer:"+" "+str(clf.score(X_scaled,y_arr)))
