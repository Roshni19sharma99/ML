''' Steps for data preprocessing
1. import pandas, numpy and matplotlib
2. read the dataet and check the data cateogory and and information and distribution
3. One hot encoding for categorical data and replace encoded data with categorical data
4. drop unwanted columns
5. fill the NA or Null values
6. split the data into training and test sets
7. Observe data distribution and do the sampling  
Feature scaling must be done after the splitting the training and test sets
standardization vs normalization
Standardization = (x-mean(x))/standard devaiation(x)   --> data is 
Normalization = x-min(x)/(max(x)-min(x))
normalization of data is prefered when the data is distributed normally
Standardization is suitable for all types of data distribution
'''

#Importing all required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler


class DataLoader():
    def __init__(self, **kwargs):
        self.data = None
    
    def load_dataset(self, path):
        self.data = pd.read_csv(path)

    def Null_value_handling(self):
        #dropping unwanted columns
        self.data.drop(["Name"], axis=1, inplace=True)
        self.data.drop(["Unnamed: 0"], axis=1, inplace=True)
        self.data.drop(["Android_version"], axis=1, inplace=True)
        self.data.drop(["Processor_name"], axis=1, inplace=True)
        self.data['Price'] = self.data['Price'].str.replace(',','')
        self.data['Price'] = self.data['Price'].astype(int)

        #handling null values for categorical data
        self.data.dropna(axis=0,inplace=True)

        #handling null value for numberical data
        '''
        
        impute = SimpleImputer(missing_values=np.nan, strategy='mean')
        impute.fit(self.data[:,1:3])
        impute.transform(self.data[:,1:3])
        '''

    def preprocess_data(self):

        #One-hot encode all categorical columns
        categorical_col = ["Rating",
                           "Spec_score",
                           "No_of_sim",
                           "Ram",
                           "Battery",
                           "Display",
                           "Camera",
                           "External_Memory",
                           "company",
                           "Inbuilt_memory",
                           "fast_charging",
                           "Screen_resolution",
                           "Processor"]
        encoded = pd.get_dummies(self.data[categorical_col],
                                 dtype=float)
        

        #Impute missing values of Android_version,Inbuilt_memory,fast_charging,Screen_resolution
        # self.data.Android_version = self.data.Android_version.fillna(0)
        
        #Update data with new columns
        self.data = pd.concat([encoded, self.data], axis=1)
        self.data.drop(categorical_col, axis=1, inplace=True)

    
    #spliting the dataset into training and test dataset
    def get_data_split(self):
        x = self.data.iloc[:,:-1]
        y = self.data.iloc[:,-1]
        return train_test_split(x,y,test_size=0.20, random_state=2021)

    #Sampling the dataset for unbalanced dataset
    def oversample(self, X_train, y_train):
        oversample = RandomOverSampler(sampling_strategy='minority')
        #convert Numpy and oversample
        x_np = X_train.to_numpy()
        y_np = y_train.to_numpy()
        x_np,y_np = oversample.fit_resample(x_np,y_np)
        #convert back to pandas
        x_over = pd.DataFrame(x_np, columns=X_train.columns)
        y_over = pd.Series(y_np, name=y_train.name)
        return x_over, y_over
    
    #Scaling the dataset with standardization,Scaling is fitting data into a particluar range
    def feature_scaling(self, X_train,X_test):
        sc = StandardScaler()
        X_train[:] = sc.fit_transform(X_train[:])
        X_test[:] = sc.transform(X_test[:])
        return X_train,X_test

