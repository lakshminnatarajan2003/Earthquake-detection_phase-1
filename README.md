# earthquake_detection_using_python_1
using ai detecting the earthquake
state your users' needs and problems
We will use random forest algorithm in this project: 

#clean the data set
In first step to execute the project we clean the data set
https://www.kaggle.com/datasets/warcoder/earthquake-dataset
The .read_csv() function takes a path to a CSV file and reads the data into a Pandas DataFrame object.
we need to find a missing values and remove null values , find the outliers   fill the correct values by using fillna() method.
We need to import the pandas library to work with the spreadsheet-like data enabling fast loading , aligning, manipulating, and merging , in addition to other key functions. 
We need to import the numpy  library to working with numerical values as it makes it easy to apply mathematical functions.
The fillna() method replaces the NULL values with a specified value.
The unique function in pandas is used to find the unique values from a series. A series is a single column of a data frame.
The cleaned data set are obtained

#data visualization
import this packages for data visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
plot the variables 
plt.bar(x,y)
plt.hist(data["x"])
plt.boxplot(x)

# create model,predict and train the model
import numpy as numpy
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import this packages for train the model

#create the model
using random forest algoritham
model=RandomForestRegressor()

#the  root mean square value given in the table
the root mean squre value is in between the 0.2 to 1.8, there earthquake will occur

#Train the model
model.fit(x,y)

#test the model
predictions=model.predict(x)

#This is how scikit-learn calculates model.score(X_test,y_test):
x = ((y_test - y_predicted) ** 2).sum()
y = ((y_test - y_test.mean()) ** 2).sum()
score = 1 - (x/y)

# make the predictions
prediction=model.predict(new_data)

the accuracy is 88%
so we use the random forest algorithm
[read me fie.docx](https://github.com/lakshminnatarajan2003/Earthquake-detection_phase-1/files/13235338/read.me.fie.docx)
