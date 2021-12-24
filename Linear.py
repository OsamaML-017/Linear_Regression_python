import pandas
import numpy
import sklearn
import seaborn

companies = pandas.read_csv("""Enter the directory where 'Companies.csv' is located""")
X=companies.iloc[:,:-2].values
y=companies.iloc[:,4].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(X_train,y_train)


y_predict=regression.predict(X_test)

from sklearn.metrics import r2_score
accuracy = r2_score(y_test,y_predict)
print('The accuracy of this Linear Regression Model is : {0}'.format(accuracy))

user_values=numpy.array([[0,0,0]])
print("Wanna check it yourself?!!, well go ahead: \n\r")
rs=float(input("Enter R&D spend of the company: "))
ads=float(input("Enter Administration spend of the company: "))
ms=float(input("Enter Marketing spend of the company: "))

user_values[0,0]=rs
user_values[0,1]=ads
user_values[0,2]=ms

user_predict=float(regression.predict(user_values))

print("The estimated profit of the company, according to the data given, is: {0}".format(user_predict))
