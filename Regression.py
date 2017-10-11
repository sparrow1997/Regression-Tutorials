%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

titanic = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

#Above dataset is available at https://www.kaggle.com/c/titanic/data

titanic.head()
titanic = titanic.drop(['PassengerId','Name','Ticket'],axis=1)
test = test.drop(['Name','Ticket'],axis=1)
titanic.isnull().sum()

max_occured = titanic['Embarked'].value_counts().idxmax()

#Embarked -> Starting Station
sns.countplot(x='Embarked', data=titanic)
#We fill S as it is the max occuring value as visible from the countplot
titanic['Embarked'] = titanic['Embarked'].fillna("S") 
sns.factorplot('Embarked','Survived',data = titanic, size = 4, aspect = 4);

#Conclusion from factor plot is survival rate of S is minimum .


#Let's check if the data in embarked is of considerable amt of stddev from the mean or not.
print("Value of all elements : ",titanic['Embarked'].value_counts())
print("Most Occured ",titanic['Embarked'].value_counts().idxmax())

#Since approximately only 25% data can be observed to exhibit some amt of variance hence
# We can consider dropping the Embarked Column

#A graphical representation for , I don't know why :)

def convert(str):
    if(str == 'S'):
        return 1
    elif(str == 'C'):
        return 2
    else :
        return 3
        
    
titanic['Embarked'] = titanic['Embarked'].apply(convert)
test['Embarked'] = test['Embarked'].apply(convert)

def convert(str):
    if(str == 'male'):
        return 1
    elif(str == 'female'):
        return 2
        
    
titanic['Sex'] = titanic['Sex'].apply(convert)
test['Sex'] = test['Sex'].apply(convert)

temp = titanic['Embarked'].value_counts()
k = []
for i in range(1,4):
    k.append(temp[i])
    
l = [1,2,3]

plt.bar(l,k)
plt.show()

#Hence proved Graphically
del titanic['Embarked']
#Fare

print(test['Fare'].isnull().sum() ,"Null Values in this column")
#Only 1 null entry
test['Fare'].fillna(test['Fare'].median(),inplace=True)

titanic['Fare'] = titanic['Fare'].astype(int)
test['Fare'] = test['Fare'].astype(int)

#Checking if This column shows enough variance to be accepted
sns.countplot(x ='Fare',data = titanic);

#Since Age Still posseses Null Values it is not plotted in pairplor
cols = ['Survived', 'Pclass', 'Sex',
         'Fare', 'Cabin']
sns.pairplot(titanic[cols])
plt.show()

titanic['Fare'].value_counts()
Since we have 91 unique values which are capable enough of no adverse effect on our predictions 
Hence we can continue using it.

#Since we have 91 unique values which are capable enough of no adverse effect on our predictions 
#Hence we can continue using it.
#We'll check if it needs to be normalised as it gives us some clues about it in the pairplot

#Age
fig,(axis1,axis2) = plt.subplots(1,2,figsize=(10,10))
average_age = titanic['Age'].mean()
std_age = titanic['Age'].std()
count_null = titanic['Age'].isnull().sum()


test_average_age = test['Age'].mean()
test_std_age = test['Age'].std()
test_count_null = test['Age'].isnull().sum()
#Checking What is more fruitfull Dropping the null values or alloting them some values
#around the mean
#Converting the age values from float to int
titanic_rand = np.random.randint(average_age-std_age,
                                 average_age+std_age,
                                 size=count_null)

test_rand = np.random.randint(test_average_age-test_std_age,
                             test_average_age+test_std_age,
                             size = test_count_null)


df = df_dummy = titanic
titanic['Age'][np.isnan(titanic['Age'])]  = titanic_rand
test['Age'][np.isnan(test['Age'])]  = test_rand


df['Age'][np.isnan(df['Age'])] = titanic_rand
df['Age'] = df['Age'].astype(int)

df_dummy['Age'] = df_dummy['Age'].dropna().astype(int)
#Comparing the value of age column in both the cases
axis1.set_title("Random Values Of Age")
axis2.set_title("Removing Null Values Of Age")
df['Age'].hist(bins = 100,ax = axis1)
df_dummy['Age'].hist(bins = 100,ax = axis2)



#Comparing the relation of Age with other features of the dataset in both the cases 
corr_mat = df.corr()
fig,ax = plt.subplots(figsize = (10,10))
sns.heatmap(corr_mat,annot = True,vmax = 1.0)

dummy_corr_mat = df_dummy.corr()
fig,ax = plt.subplots(figsize = (10,10))
sns.heatmap(corr_mat,annot = True,vmax = 1.0)

df_cols = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin']
dummy_cols = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin']

sns.pairplot(df[df_cols])
plt.show()

sns.pairplot(df_dummy[dummy_cols])
plt.show()

titanic['Age'][np.isnan(titanic['Age'])] = titanic_rand

print(titanic['Cabin'].isnull().sum())
# Since it has a lot of null values we prefer deleting it
titanic.drop('Cabin',axis =1,inplace =True)
test.drop('Cabin',axis =1,inplace =True)

#Age

# Instead of having two columns Parch & SibSp, 
# we can have only one column represent if the passenger had any family member aboard or not,
# Meaning, if having any family member(whether parent, brother, ...etc) will increase chances of Survival or not.

titanic['Family'] = titanic['SibSp']+titanic['Parch']
del titanic['SibSp']
del titanic['Parch']

titanic['Family'].loc[titanic['Family']>0] = 1
titanic['Family'].loc[titanic['Family'] == 0] = 0

test['Family'] = test['SibSp']+test['Parch']
del test['SibSp']
del test['Parch']

test['Family'].loc[test['Family']>0] = 1
test['Family'].loc[test['Family'] == 0] = 0

#Just a visual Description of Effect Of having a family on Survival Rate
sns.factorplot('Family','Survived',data = titanic,size = 4,aspect =4);
#Hence the ones with family had high chances of survival

#Sex 

from scipy.stats import norm
# As we saw in the above heatmaps , sex is highly correlated to Survived so there is no point of removing it
sns.countplot(x='Sex',data = titanic)
sns.factorplot('Sex','Survived',data = titanic,size = 4,aspect =4);

fig= sns.kdeplot(titanic['Sex'],titanic['Survived'] ,shade=True)

#Looks like feminism was quite effective in that time , lol :P

#PClass

sns.factorplot('Pclass','Survived',order = [1,2,3],data = titanic,size = 5);

#So the riches and the royals were the first one to be rescued...Hmmmmm...Let's see what all can this
#tell us about the case

#Well ,only travel in a ship If YOU ARE RICH...increases your chances of survival :p

pclass_dummies = pd.get_dummies(titanic['Pclass'])
pclass_dummies_test = pd.get_dummies(test['Pclass'])

pclass_dummies.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']

del titanic['Pclass']
del test['Pclass']

titanic = titanic.join(pclass_dummies)
test = test.join(pclass_dummies_test)

titanic.head()

xtrain = titanic.drop('Survived',axis = 1)
ytrain = titanic['Survived']
xtest = test.drop('PassengerId',axis = 1)

del xtest['Embarked']

#Logistic Regression

lr = clf  = LogisticRegression()
clf.fit(xtrain,ytrain)
y_pred = clf.predict(xtest)
clf.score(xtrain,ytrain)

svc_l = clf  = SVC(kernel = 'linear')
clf.fit(xtrain,ytrain)
y_pred = clf.predict(xtest)
clf.score(xtrain,ytrain)

svc_g = clf  = SVC(kernel = 'rbf')
clf.fit(xtrain,ytrain)
y_pred = clf.predict(xtest)
clf.score(xtrain,ytrain)

rf = clf  = RandomForestClassifier(n_estimators = 500)
clf.fit(xtrain,ytrain)
rf_y_pred = clf.predict(xtest)
clf.score(xtrain,ytrain)
gsn = clf  = GaussianNB()
clf.fit(xtrain,ytrain)
y_pred = clf.predict(xtest)
clf.score(xtrain,ytrain)

knc = clf  = KNeighborsClassifier(n_neighbors = 3)
clf.fit(xtrain,ytrain)
y_pred = clf.predict(xtest)
clf.score(xtrain,ytrain)





