

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
pd.set_option('display.max_columns',None)
burnoutdf = pd.read_csv('C:/Users/sheri/OneDrive/Desktop/employee_burnout_analysis-AI.csv')
burnoutdf

"""# New Section"""

burnoutdf['Date of Joining']=pd.to_datetime(burnoutdf['Date of Joining'])

burnoutdf.shape

burnoutdf.head()

burnoutdf.info()



burnoutdf.tail()

burnoutdf.duplicated().sum()

burnoutdf.isna().sum()



burnoutdf.columns

burnoutdf.describe()

for i,col in enumerate(burnoutdf.columns):
  print(f"/n/n{burnoutdf[col].unique()}")
  print(f"/n{burnoutdf[col].value_counts()}\n\n")



burnoutdf=burnoutdf.drop(['Employee ID'],axis=1)

intFloatburnoutdf=burnoutdf.select_dtypes([np.int, np.float])
for i, col in enumerate(intFloatburnoutdf.columns):
  if (intFloatburnoutdf[col].skew() >= 0.1):
    print("\n",col, "feature is Positively skewed and value is: ", intFloatburnoutdf[col].skew())
  elif (intFloatburnoutdf[col].skew() <= -0.1):
    print("\n",col, "feature is Negtively Skewed and value is: ", intFloatburnoutdf[col].skew())
  else:
    print("\n",col, "feature is Normally Distributed and value is: ", intFloatburnoutdf[col].skew())

burnoutdf['Resource Allocation'].fillna(burnoutdf['Resource Allocation'].mean(), inplace=True)
burnoutdf[ 'Mental Fatigue Score'].fillna (burnoutdf['Mental Fatigue Score'].mean(),inplace=True)
burnoutdf[ 'Burn Rate'].fillna(burnoutdf[ 'Burn Rate'].mean(),inplace=True)

burnoutdf.isna().sum()

burnoutdf.corr()

Corr=burnoutdf.corr()
sns.set(rc={'figure.figsize' :(14,12)})
fig=px.imshow(Corr, text_auto=True, aspect="auto")
fig.show()

plt.figure(figsize=(10,8))
sns.countplot(x="Company Type", data=burnoutdf, palette="Spectral")
plt.title("Plot Distribution of Company Type")
plt.show()

plt.figure(figsize=(10,8))

sns.countplot(x="WFH Setup Available", data=burnoutdf, palette="dark:salmon_r")
plt.title("Plot Distribution of WFH_Setup_Available")

plt.show()

burn_st=burnoutdf.loc[:, 'Date of Joining': 'Burn Rate']
burn_st-burn_st.select_dtypes([int, float])
for i, col in enumerate(burn_st.columns):
  fig = px.histogram(burn_st, x=col, title="Plot Distribution of "+col, color_discrete_sequence=["indianred"])
  fig.update_layout(bargap=0.2)
  fig.show()

fig = px.line(burnoutdf, y="Burn Rate", color="Designation", title="Burn rate on the basis of Designation",color_discrete_sequence=px.colors.qualitative.Pastel1)

fig.update_layout(bargap=0.1)
fig.show()

fig = px.line(burnoutdf, y="Burn Rate", color="Gender", title="Burn rate on the basis of Gender",color_discrete_sequence=px.colors.qualitative.Pastel1)

fig.update_layout(bargap=0.1)
fig.show()

fig = px.line(burnoutdf, y="Mental Fatigue Score", color="Designation", title="Mental Fatigue vs designation",color_discrete_sequence=px.colors.qualitative.Pastel1)

fig.update_layout(bargap=0.2)
fig.show()

sns.relplot(

     data=burnoutdf, x="Designation", y="Mental Fatigue Score", col="Company Type",
     hue="Company Type", size="Burn Rate", style="Gender",
     palette=["g","r"], sizes=(50,200)
  )

from sklearn import preprocessing
Label_encode=preprocessing.LabelEncoder()

#Assign in new variable

burnoutdf['GenderLabel']=Label_encode.fit_transform(burnoutdf['Gender'].values)
burnoutdf[ 'Company TypeLabel']=Label_encode.fit_transform(burnoutdf['Company Type'].values)
burnoutdf['WFH Setup Available']=Label_encode.fit_transform(burnoutdf['WFH Setup Available'].values)

gn= burnoutdf.groupby('Gender')
gn= gn['GenderLabel']
gn.first()

ct = burnoutdf.groupby('Company Type')
ct = ct['Company TypeLabel']
ct.first()

wsa = burnoutdf.groupby('WFH Setup Available')
wsa = wsa['WFH Setup Available']
wsa.first()

columns=['Designation', 'Resource Allocation','Mental Fatigue Score','GenderLabel', 'Company TypeLabel', 'WFH Setup Available']
X= burnoutdf[columns]
y=burnoutdf['Burn Rate']

print(X)

print(y)

from sklearn.decomposition import PCA
pca=PCA(0.95)
x_pca = pca.fit_transform(X)
print("PCA shape of x is: ",x_pca.shape, "and orignal shape is: ", X.shape)
print("% of importance of selected features is:", pca.explained_variance_ratio_)
print("The number of features selected through PCA is:", pca.n_components_)

from sklearn.model_selection import train_test_split
x_train_pca,x_test,Y_train,Y_test = train_test_split(x_pca,y,test_size=0.75,random_state=15)

print(x_train_pca.shape,x_test.shape,Y_train.shape,Y_test.shape)

from sklearn.metrics import r2_score

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor()

rf_model.fit(x_train_pca,Y_train)

train_pred_rf = rf_model.predict(x_train_pca)
train_r2= r2_score(Y_train, train_pred_rf)
test_pred_rf = rf_model.predict(x_test)
test_r2 =r2_score(Y_test, test_pred_rf)
print("Accuracy score of train data: "+str(round(100*train_r2,4))+" %")
print("Accuracy score of test data: "+str(round(100*test_r2,4))+" %")

from sklearn.ensemble import AdaBoostRegressor
abr_model = AdaBoostRegressor()
abr_model.fit(x_train_pca, Y_train)

train_pred_adboost = abr_model.predict(x_train_pca)
train_r2 = r2_score(Y_train, train_pred_adboost)
test_pred_adaboost = abr_model.predict(x_test)
test_r2 = r2_score(Y_test, test_pred_adaboost)
print("Accuracy score of train data: "+str(round(100*train_r2,4))+" %")
print("Accuracy score of test data: "+str(round(100*test_r2,4))+" %")
