import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import pylab
import scipy.stats as state

df = pd.read_csv("Ecommerce.csv")
a=df.head()
print(a)
b=df.info()
print(b)
c=df.describe()
print(c)


# this plot show same relationship 
# sns.jointplot(x="Time on App",y="Yearly Amount Spent",data=df, alpha=0.3) 

# this shows all relationship with all veriables
# sns.pairplot(df, kind='scatter',plot_kws={'alpha':0.4})


# This show max relationship
# sns.jointplot(x="Length of Membership",y="Yearly Amount Spent",data=df, alpha=0.3) 


#  This is linearRegression plot
# sns.lmplot(x="Length of Membership", y="Yearly Amount Spent", data=df, scatter_kws={'alpha':0.3})


# plt.show()

x_ =df[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]
y_=df['Yearly Amount Spent']

x_train, x_test ,y_train, y_test = train_test_split(x_,y_,test_size=0.3, random_state=73)

# Training the model

lm = LinearRegression()

lm.fit(x_train,y_train)

lm.coef_ # this give the coef of varible which is more imp to find unnone variable in this case that is yearly ammount spent

#prediction predict the value of y  for x_test which aprox should be equal to y_test
prediction = lm.predict(x_test)
# print(prediction)

# sns.scatterplot(x=prediction ,y=y_test)
# plt.xlabel("predictions")

# plt.show()

# print("mean absolute error: ",mean_absolute_error(y_test,prediction))
# print("mean squared error: ",mean_squared_error(y_test,prediction))
# print("RMSR: ",math.sqrt(mean_squared_error(y_test,prediction)))

residuals = y_test-prediction # distance between  y_test and prediction
# print(residuals)

# sns.displot(residuals, bins=50, kde=True)
# plt.show()

state.probplot(residuals,dist='norm',plot=pylab)
pylab.show()