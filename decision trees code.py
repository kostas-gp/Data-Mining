# import the libraries
import csv
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.externals import joblib

#replace the missing value with the medium
variable = []
temp = 0
count =0
with open('C:\\Users\\konstantinos\\Desktop\\train_final.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        if row[19] != 'NaN':
            temp = temp + float(row[19])
            count = count + 1
        variable.append(row)

for i in range(len(variable)):
    for j in range(len(variable[i])):
         if variable[i][j] == 'NaN':
             variable[i][j] = temp/count

f = open('C:\\Users\\konstantinos\\Desktop\\train_final.csv', "w+")
f.close()

with open('C:\\Users\\konstantinos\\Desktop\\train_final.csv', 'ab') as file:
    writer = csv.writer(file)
    writer.writerows(variable)

# load dataset
balance_data = pd.read_csv('C:\\Users\\konstantinos\\Desktop\\train_final.csv', sep=',', header=None)

# save the data
X = balance_data.values[:, 0:19] 
Y = balance_data.values[:, 20]
Y=Y.astype('int')

# Splitting the dataset into train and test 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(  
X, Y, test_size = 0.3, random_state = 100) 

# Create Decision Tree classifer
model = DecisionTreeClassifier()

# Train Decision Tree Classifer
model = model.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = model.predict(X_test)

# print the accurancy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#replace the missing value with the medium
variable = []
temp = 0
count =0
with open('C:\\Users\\konstantinos\\Desktop\\test.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        if row[19] != 'NaN':
            temp = temp + float(row[19])
            count = count + 1
        variable.append(row)

for i in range(len(variable)):
    for j in range(len(variable[i])):
         if variable[i][j] == 'NaN':
             variable[i][j] = temp/count

f = open('C:\\Users\\konstantinos\\Desktop\\test.csv', "w+")
f.close()

with open('C:\\Users\\konstantinos\\Desktop\\test.csv', 'ab') as file:
    writer = csv.writer(file)
    writer.writerows(variable)

# save the model to disk
filename = 'C:\\Users\\konstantinos\\Desktop\\create_model.sav'
joblib.dump(model, filename)

# load the model from disk
loaded_model = joblib.load(filename)

# load dataset from the test
balance_data = pd.read_csv('C:\\Users\\konstantinos\\Desktop\\test.csv', sep=',', header=None)

# save the data
X_test = balance_data.values[:, 0:19] 

result = model.predict(X_test)
#save the prediction in test csv
var = []
index = 0
with open('C:\\Users\\konstantinos\\Desktop\\test.csv', 'r') as file:
    reader2 = csv.reader(file)
    for row in reader2:
       row[20] = result[index]
       index = index + 1
       var = row
f = open('C:\\Users\\konstantinos\\Desktop\\test.csv', "w+")
f.close()

with open('C:\\Users\\konstantinos\\Desktop\\test,csv', 'ab') as file:
    writer = csv.writer(file)
    writer.writerows(var)