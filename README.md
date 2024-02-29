# Exno:1
# 1) Data Cleaning Process

# AIM
To read the given data and perform data cleaning and save the cleaned data to a file.

# Explanation
Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information.

# Algorithm
STEP 1: Read the given Data

STEP 2: Get the information about the data

STEP 3: Remove the null values from the data

STEP 4: Save the Clean data to the file

STEP 5: Remove outliers using IQR

STEP 6: Use zscore of to remove outliers

# Program
# For Data_set
```
 import pandas as pd
 df=pd.read_csv("Data_set .csv")
 df
 df.head()
 df.tail()
 df.info()
 df.isnull()
 df.isnull().sum()
 df.describe()
 df.dropna(axis=1)
 df.dropna(axis=0)
 df.fillna(0)

```
# For loan_set
```
 import pandas as pd
 df=pd.read_csv("Loan_Data.csv")
 print(df)
 df.head(5)
 df.info()
 df.isnull()
 df.isnull().sum()
 df['Loan_ID']=df['Loan_ID'].fillna(df['Education'].mode()[0])
 df['Education']=df['Education'].fillna(df['Education'].mode()[0])
 df['Married']=df['Married'].fillna(df['Education'].mode()[0])
```
```
 df.head()
 df['ApplicantIncome']=df['ApplicantIncome'].fillna(df['ApplicantIncome'].mean())
 df['LoanAmount']=df['LoanAmount'].fillna(df['LoanAmount'].mean())
 df.head()
 df.head()
 df['Loan_Amount_Term']=df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())
 df.head()
 df.info()
 df.isnull().sum()
```
 # Output
 # For Data_Set
 ![alt text](image-1.png)
 ![alt text](image-2.png)
 ![alt text](image-3.png)
 ![alt text](image-4.png)
 ![alt text](image-17.png)
 ![alt text](image-20.png)
 ![alt text](image-5.png)
 ![alt text](image-6.png)
 ![alt text](image.png)

# 2) Outlier Detection and Removal

# AIM:
To read a given dataset and remove outliers and save a new dataframe.

# ALGORITHM:
(1) Remove outliers using IQR

(2) After removing outliers in step 1, you get a new dataframe.

(3) use zscore of 3 to remove outliers. This is quite similar to IQR and you will get exact same result

(4) for the data set height_weight.csv find the following

    (a) Using IQR detect weight outliers and print them

    (b) Using ZScore, detect height outliers and print them

# PROGRAM
# IQR:
```
import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats
df = pd.read_csv("heights.csv")
sns.boxplot(data=df)
sns.scatterplot(data=df)
max =df['height'].quantile(0.90)
min =df['height'].quantile(0.15)
max
min
dq = df[((df['height']>=min)&(df['height']<=max))]
dq
low = min-1.5*iqr
high = max+1.5*iqr
dq = df[((df['height']>=min)&(df['height']<=max))]
dq
```
# ZSCORE:
```
import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats
data = {'weight':[12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63,66,69,202,72,75,78,81,84,232,87,90,93,96,99,258]}
df = pd.DataFrame(data)
df
sns.boxplot(data=df)
z = np.abs(stats.zscore(df))
print(df[z['weight']>3])
val =[12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63,66,69,202,72,75,78,81,84,232,87,90,93,96,99,258]
out=[]
def d_o(val):
    ts=3
    m=np.mean(val)
    sd=np.std(val)
    for i in val:
        z=(i-m)/sd
        if np.abs(z)>ts:
            out.append(i)
    return out
op = d_o(val)
op
```
# OUTPUT
# IQR
![alt text](image-7.png)
![alt text](image-8.png)
![alt text](image-9.png)
![alt text](image-10.png)
![alt text](image-11.png)
![alt text](image-12.png)
![alt text](image-13.png)
![alt text](image-14.png)

# Z SCORE
![alt text](image-15.png)
![alt text](image-16.png)
![alt text](image-18.png)
![alt text](image-19.png)

# Result
Thus, the given data is read, cleansed and the cleaned data is saved into the file and the given data is read,remove outliers and save a new dataframe was created and executed successfully.

