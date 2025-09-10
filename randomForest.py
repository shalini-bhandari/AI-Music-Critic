import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

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

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

print("Feature scaling complete.")

# 1. create model instance
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 2. train the model using training data
model.fit(X_train_scaled, y_train)
print("Model training complete.")

# 3. make predictions using validation data
y_preds = model.predict(X_valid_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_valid, y_preds)
print(f"Model accuracy with Random Forest: {accuracy * 100:.2f}%")

# Load the final test dataset
df_test = pd.read_csv('data/test.csv')
X_final_test = df_test[features_to_use]

X_final_test_imputed = imputer.transform(X_final_test)
X_final_test_scaled = scaler.transform(X_final_test_imputed)

final_predictions = model.predict(X_final_test_scaled)
print("Predictions for the final test set are complete!")

# Create the dictionary to map genre numbers to names
genre_mapping = {
    0: 'Alternative',
    1: 'Anime',
    2: 'Blues',
    3: 'Classical',
    4: 'Country',
    5: 'Electronic',
    6: 'Hip-Hop',
    7: 'Jazz',
    8: 'Rap',
    9: 'Rock',
    10: 'Pop'
}

print("Genre mapping dictionary created.")

# Create a new DataFrame with the song info and our predictions
results_df = pd.DataFrame({
    'Artist Name': df_test['Artist Name'],
    'Track Name': df_test['Track Name'],
    'Predicted Genre': final_predictions
})
results_df['Predicted Genre Name'] = results_df['Predicted Genre'].map(genre_mapping)

# Save the results to a CSV file
results_df.to_csv('results/my_music_predictions.csv', index=False)

print("\nResults file 'my_music_predictions.csv' has been created.")
print(results_df.head())
# Get the details of the first song in the test set
first_song_artist = df_test.iloc[0]['Artist Name']
first_song_track = df_test.iloc[0]['Track Name']

# Get the predicted genre for that song
predicted_genre = results_df.iloc[0]['Predicted Genre']

# You might need a simple dictionary to map the genre number back to a name
# Let's assume you have a list of genre names from your EDA step
# For example: genre_mapping = {0: 'Rock', 1: 'Indie', ...}
# If not, we can just use the number for now.

print("\n--- AI Music Critic Review ---")
print(f"Track: '{first_song_track}' by {first_song_artist}")
print(f"Verdict: After analyzing its audio features, my model classifies this track as genre number {predicted_genre}.")