

#libraries


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




#data preprocessing

veriler = pd.read_csv("eksikveriler.csv")

print(veriler)


#Solution for missed variables
#taking mean


#sci-kit learn

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
Yas = veriler.iloc[:,1:4].values
print(Yas)

imputer = imputer.fit(Yas[:,1:4]) 
#Fit for learning we want to learn 1 to 4 column in yas, learning mean
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)
# nal values replace to 28.45 mean value
# fit learn transform apply

# we try to order country like 0 tr, 1 us, 2 fr

ulke = veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
print(ulke)


ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray() 

print(ulke)


#we make 2 different dataframe seperatly one of about country one of other parameters
sonuc = pd.DataFrame(data = ulke , index = range(22),columns = ["fr","tr","us"])

print(sonuc)

sonuc2 = pd.DataFrame(data = Yas, index = range(22), columns =["boy","kilo","yas"])
print(sonuc2)

# there is another dataframe about gender

cinsiyet = veriler.iloc[:,-1].values

print(cinsiyet)

sonuc3 = pd.DataFrame(data = cinsiyet , index = range(22), columns = ["cinsiyet"])

print(sonuc3)


#Combining dataframes

s = pd.concat([sonuc,sonuc2],axis = 1) # axis for ordering,colum or row
print(s)

s2 = pd.concat([s,sonuc3], axis = 1)
print(s2)

# data split for test and train


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(s,sonuc3,test_size = 0.33, random_state = 0)


#converting 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)





