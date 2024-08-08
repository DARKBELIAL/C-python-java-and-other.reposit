# Import TensorFlow and check version
import tensorflow as tf
print(tf.__version__)

# Importing basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
dataset.head()

# Extract features and target variable
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

# One-Hot Encoding for categorical features
geography = pd.get_dummies(X['Geography'], drop_first=True)
gender = pd.get_dummies(X['Gender'], drop_first=True)

# Concatenate the original DataFrame with the one-hot encoded columns
X = pd.concat([X, geography, gender], axis=1)

# Drop the original categorical columns
X = X.drop(['Geography', 'Gender'], axis=1)

# Splitting the dataset into training set and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Creating ANN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

classifier = Sequential()

# Input layer and first hidden layer
classifier.add(Dense(units=11, activation='relu', input_dim=X_train.shape[1]))

# Second hidden layer
classifier.add(Dense(units=7, activation='relu'))
classifier.add(Dropout(0.2))

# Third hidden layer
classifier.add(Dense(units=6, activation='relu'))
classifier.add(Dropout(0.3))

# Output layer
classifier.add(Dense(units=1, activation='sigmoid'))

# Compile the model
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
classifier.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=20,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=True  # Change to True to restore best weights
)

# Train the model
model_history = classifier.fit(X_train, y_train, validation_split=0.33, batch_size=11, epochs=1000, callbacks=[early_stopping])

# Plotting the loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('Model Loss Progress During Training')
plt.ylabel('Training and Validation Loss')
plt.xlabel('Epoch Number')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()

# Predicting the test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Accuracy score
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
print(score)



---------------------------------------------------------------------------------------------------------------------------



# Function to preprocess single customer data and predict churn
def preprocess_and_predict(data, classifier, sc):
    # Create a DataFrame with the same columns as the training data
    columns = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    df = pd.DataFrame([data], columns=columns)

    # One-Hot Encoding for categorical features
    geography = pd.get_dummies(df['Geography'], drop_first=True)
    gender = pd.get_dummies(df['Gender'], drop_first=True)

    # Reindex the one-hot encoded DataFrames to include all possible columns
    geography = geography.reindex(columns=['Germany', 'Spain'], fill_value=0)
    gender = gender.reindex(columns=['Male'], fill_value=0)

    # Concatenate the original DataFrame with the one-hot encoded columns
    df = pd.concat([df, geography, gender], axis=1)

    # Drop the original categorical columns
    df = df.drop(['Geography', 'Gender'], axis=1)

    # Feature scaling
    df = sc.transform(df)

    # Predicting the result
    y_pred = classifier.predict(df)
    return (y_pred > 0.5)

# Example test data
test_data = [600, 'France', 'Male', 40, 3, 60000, 2, 1, 1, 50000]
result = preprocess_and_predict(test_data, classifier, sc)
print("Will the customer churn?:", result[0][0])
