import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

file_path = 'data/train.csv' 
df_train = pd.read_csv(file_path)

print(df_train.head())
print(df_train.info())

# The 'Class' column is what we want to predict. Let's rename it to 'music_genre' for clarity.
df_train.rename(columns={'Class': 'music_genre'}, inplace=True)

# Define the columns we want to use as features
# We are excluding the track/artist names and the target variable itself
features_to_use = [
    'Popularity', 'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'duration_in min/ms', 'time_signature'
]

# Create our feature set X and target y
X = df_train[features_to_use]
y = df_train['music_genre']

# Let's check the first few rows of our new feature set X
print("Our Features (X):")
print(X.head())

# Create an imputer object that fills missing values with the column's mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=features_to_use)

print("\nMissing values after imputation:")
print(X.isnull().sum())

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

print(f"\nTraining set size: {X_train.shape}, Validation set size: {X_valid.shape}")
print(f"Training labels size: {y_train.shape}, Validation labels size: {y_valid.shape}")

# 1. create model instance
model = DecisionTreeClassifier(random_state = 42)

# 2. train the model using training data
model.fit(X_train, y_train)

print("Model training complete.")

# 3. make predictions using validation data
y_preds = model.predict(X_valid)
print("Predictions on validation set complete.")
print(f"Predictions: {y_preds}")

# 4. evaluate the model
accuracy = accuracy_score(y_valid, y_preds)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_valid, y_preds)

# Visualize the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)

plt.title('Confusion Matrix')
plt.ylabel('Actual Genre')
plt.xlabel('Predicted Genre')
plt.show()