import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
df_train = pd.read_csv('../input/train.csv')
df_train.columns
df_train.describe()
df_train['SalePrice'].describe()
sns.distplot(df_train['SalePrice']);
#It means that around 80% values are evaluated at 180921(approx)(i.e. the mean)
#Skewness means degree of Difference from being a normal curve  if  Positive Skewness
#The graph's reverse bell would be displaced towards the left.
# i.e. Mode<Median<Mean

#Kurtosis means degree of Difference in the bell shape i.e too peaked or too fat
#In our case it is leptokurtosis

print("Skewness  %f" % df_train['SalePrice'].skew())
print("Kurtosis  %f" % df_train['SalePrice'].kurt())

corrmat = df_train.corr()
f,ax = plt.subplots(figsize=(12,12))
sns.heatmap(corrmat, vmax=0.8);

****Conclusions Drawn from the HeatMap state that:

1.  OverallQual,TotalBsmtFinSF1,GrLivArea,GrageCars,GarageArea are  in a" Strong Positive Correlation"  with  SalesPrice(i.e. Feature - Label Correlation)

2. The 2 "Big Squares" represent IntraFeature Correlation.
     Such Features effect the prediction model in a similar model.
     Hence it is adviced to use anyone of such features.
     

k=12

cols = corrmat.nlargest(k,'SalePrice')['SalePrice'].index
#Finding the index of top 11 most correlated Elements with SalePrice
#i.e. the first row and the first col contains the top 11 values.

cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale = 1.0)
hm = sns.heatmap(cm,cbar = True,
                 annot = True, square = True , 
                 fmt='.1f',
                 annot_kws = {'size':8},
                 yticklabels=cols.values,
                 xticklabels=cols.values)
plt.show()

Taking in consideration another form of plasma soup for highly correlated features(with label)
Note: 'Most Corelated Features' are obtained from the heatmap.

## PairPlots

sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars',
        'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols],size = 2.0)
plt.show()

Plotting the highly Correlated Features Individually

sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars',
        'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols],size = 2.0)
plt.show()


Plotting the highly Correlated Features Individually
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'],df_train[var]],axis = 1)

data.plot.scatter(x = var,y ='SalePrice',ylim = (0,800000))
data.plot()

var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'],df_train[var]],axis = 1)

data.plot.scatter(x = var,y ='SalePrice',ylim = (0,800000))
data.plot()

var = 'OverallQual'
data = pd.concat([df_train['SalePrice'],df_train[var]],axis = 1)

data.plot.scatter(x = var,y ='SalePrice',ylim = (0,800000))
data.plot()

var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'],df_train[var]],axis = 1)

data.plot.scatter(x = var,y ='SalePrice',ylim = (0,800000))
data.plot()

#box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);

#box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f,ax = plt.subplots(figsize = (6,8))
fig = sns.boxplot(x = var, y="SalePrice", data=data)
fig.axis(ymin = 0,ymax = 800000);

var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f,ax = plt.subplots(figsize = (6,8))
fig = sns.boxplot(x = var, y="SalePrice", data=data)
fig.axis(ymin = 0,ymax = 800000);

# Why not df_train.isnull().count()?
# Because It returns the 'count of number of rows/col it traversed'
#and sum() returns the sum pf all trues by considering True = 1
df_train.isnull().sum().sort_values(ascending = False)

total = df_train.isnull().sum().sort_values(ascending = False)
percentage = (df_train.isnull().sum()/ df_train.isnull().count()).sort_values(ascending = False)
null_data = pd.concat([total,percentage],axis = 1, keys = ['Total' , 'Percent'])
null_data.head(30)

## Let's Remove the Features with more than 15% Null Values 

df_train = df_train.drop((null_data[null_data['Percent']>15]).index,1)

## What about the remaining entries with less than 15% null?

### Well, we will examine their importance using heatmap ,boxplots,pairplots etc.

### Specifically in this Dataset the above mentioned entries with less than 15% null
### are not of such importance hence we can remove them too.

#Checking if any Null value is left
df_train.isnull().sum().max()


** Bivariate Analysis
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'],df_train[var]],axis = 1)
data.plot.scatter(x = var, y = 'SalePrice', ylim = (0,800000))

The two rightmost points are exceptions and can result in creating a mess in the model although their occurence is not so significant in quantity but on a safer side we ignore them.

In a real world context these 2 points might depict land with low "Overall Quality" or other features might be negatively correlated with "SalePrice"

df_train.sort_values(by = 'GrLivArea', ascending = True)

df_train =df_train[(df_train.GrLivArea<4500)]

var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'],df_train[var]],axis = 1)
data.plot.scatter(x = var, y = 'SalePrice', ylim = (0,800000))

Other Correlated Features Can be Analysed in the similar manner

df_train.sort_values(by = 'GrLivArea', ascending = True)
df_train =df_train[(df_train.GrLivArea<4500)]
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'],df_train[var]],axis = 1)
data.plot.scatter(x = var, y = 'SalePrice', ylim = (0,800000))

## Normalization in Depth

# Let's take in regard the 'SalePrice'
sns.distplot(df_train['SalePrice'],fit = norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'],plot = plt)

To remove Positive Skewness we can use 'Log' 
And Plot Again

df_train['SalePrice'] = np.log(df_train['SalePrice'])
sns.distplot(df_train['SalePrice'],fit = norm)
fig = plt.figure()
plotter = stats.probplot(df_train['SalePrice'],plot =plt)
#Similarly for GrLivArea

sns.distplot(df_train['GrLivArea'],fit = norm);
fig = plt.figure()
diagram = stats.probplot(df_train['GrLivArea'],plot = plt)
# This one also needs a bit 'Logistoic magic'
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
sns.distplot(df_train['GrLivArea'],fit = norm);
fig = plt.figure()
diagram = stats.probplot(df_train['GrLivArea'],plot = plt)

df_train['SalePrice'] = np.log(df_train['SalePrice'])
sns.distplot(df_train['SalePrice'],fit = norm)
fig = plt.figure()
plotter = stats.probplot(df_train['SalePrice'],plot =plt)
#Similarly for GrLivArea

sns.distplot(df_train['GrLivArea'],fit = norm);
fig = plt.figure()
diagram = stats.probplot(df_train['GrLivArea'],plot = plt)
# This one also needs a bit 'Logistoic magic'
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
sns.distplot(df_train['GrLivArea'],fit = norm);
fig = plt.figure()
diagram = stats.probplot(df_train['GrLivArea'],plot = plt)

### Welcome The Boss
sns.distplot(df_train['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)
### What's so special?

At certain points it has the blessed value '0'
and log 0 = Infinity
So we can't use the previous concepts.
Hence, we conclude by following the path demonstrated in the following lines:
    We will only be applying 'Log' to non zero values.
Just pay attention to the fact why we are using pd.series instead of an arra or numpy-array

So that 'loc' can be used for carrying on this process
df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0 
df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1
df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
