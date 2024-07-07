import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Aim is to predict the marks of students of the test data
# Use the file namd 'training data' to train the model
data = pd.read_excel('Training data.xlsx')
x_train = np.array(data.iloc[:,0:8])
y_train = np.array(data.iloc[:,8]).reshape(-1,1)
# Try plotting y_train with different features
# for i in range(8):
#     plt.figure()
#     plt.scatter(x_train[:, i], y_train)
#     plt.xlabel(f'Feature {i+1}')
#     plt.ylabel('Marks')
#     plt.title(f'Plot of y_train vs Feature {i+1}')
#     plt.show()

# To get an idea whether to add some features or not
# Add some features if required in x_train

# Also do label encoding for features not represented in numbers
# refer the link if not know : https://youtu.be/589nCGeWG1w?si=t2Wa7LgbUOO4RooM

def feature_changing(x_train):
    # Perform label encoding for features not represented in numbers
    label_encoder = LabelEncoder()
    for i in range(2):
    # for i in range(x_train.shape[1]):
        if not np.issubdtype(x_train[:, i].dtype, np.number):
            x_train[:, i] = label_encoder.fit_transform(x_train[:, i])
    return x_train

x_train = feature_changing(x_train)
def z_score(x_train):
    x_std = np.std(x_train, axis=0)
    x_mean = np.mean(x_train, axis=0)
    x_train = ((x_train - x_mean )/ x_std)

  # ---------
    # write the code for feature scaling here
    # Your code here
  # ---------

    return x_train,x_std,x_mean
def cost(x_train,y_train,w,b):

  # ---------
    # Your code here
    # Use mean square error as cost function
    # return cost
  # ---------
    loss = np.mean((np.dot(x_train, w) + b - y_train) ** 2)
    return loss
def gradient_descent(x_train,y_train,w,b):

  # ---------
    # Your code here
    # Choose learning rate yourself
  # ---------
    learning_rate = 0.001
    predictions = np.dot(x_train, w) + b
    errors = predictions - y_train
    dw = np.mean(errors * x_train, axis=0).reshape(-1, 1)
    db = np.mean(errors)
    w -= learning_rate * dw
    b -= learning_rate * db
    return w, b
x_train = x_train.astype(np.float64)
x_train,x_std,x_mean = z_score(x_train)

np.random.seed(2147483647)
w = np.random.randn(x_train.shape[1],1)
b = np.random.randn(1)

old_cost = 0

while abs(old_cost - cost(x_train,y_train,w,b))>0.00001:
  old_cost = cost(x_train,y_train,w,b)
  w,b = gradient_descent(x_train,y_train,w,b)

x_predict = pd.read_excel('Test data.xlsx').iloc[:,:8].to_numpy()
x_predict = feature_changing(x_predict)
x_predict = (x_predict - x_mean)/x_std
ans = pd.read_excel('Test data.xlsx').iloc[:,8].to_numpy()

y_predict = np.dot(x_predict,w) + b

accuracy = 0
for dim in range(len(ans)):
  if abs(y_predict[dim]-ans[dim])<0.5: # do not change the tolerance as you'll be checked on +- 0.5 error only
    accuracy += 1
accuracy = round(accuracy*100/200.0,2)
ok = 'Congratulations' if accuracy>95 else 'Optimization required'
print(f"{ok}, your accuracy is {accuracy}%")