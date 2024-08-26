import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('titanic.csv', delimiter=';', decimal=',')
data['Age'] = data['Age'].astype(float)

# Preprocessing
categorical_features = ['Pclass', 'Sex', 'Embarked']
numeric_features = ['Age', 'SibSp', 'Parch']

# Create transformers for numerical and categorical data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine transformers into a preprocessor with ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Prepare target (Survived) and features
X = data.drop('Survived', axis=1)
y = data['Survived']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Preprocess the data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Neural network model
model = Sequential()
model.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))  # First hidden layer
model.add(Dense(8, activation='relu'))  # Second hidden layer
model.add(Dense(8, activation='relu'))  # Third hidden layer
model.add(Dense(1, activation='sigmoid'))  # Output layer

print('Compiling...')
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mse'])

print('Training...')
# Train model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=10, verbose=0)

print('Creating a graph...')
# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Accuracy plot
ax1.plot(history.history['accuracy'])
ax1.plot(history.history['val_accuracy'])
ax1.set_title('Model Accuracy')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend(['Train', 'Validation'], loc='upper left')

# MSE plot
ax2.plot(history.history['mse'])
ax2.plot(history.history['val_mse'])
ax2.set_title('Model MSE')
ax2.set_ylabel('MSE')
ax2.set_xlabel('Epoch')
ax2.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()
