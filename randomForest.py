import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

file_path = 'data/train.csv' 
df_train = pd.read_csv(file_path)

print(df_train.head())
print(df_train.info())

df_train.rename(columns={'Class': 'music_genre'}, inplace=True)

features_to_use = [
    'Popularity', 'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'duration_in min/ms', 'time_signature'
]

X = df_train[features_to_use]
y = df_train['music_genre']

print("Our Features (X):")
print(X.head())

#imputer object that fills missing values with the column mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=features_to_use)

print("\nMissing values after imputation:")
print(X.isnull().sum())

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
print(f"\nTraining set size: {X_train.shape}, Validation set size: {X_valid.shape}")
print(f"Training labels size: {y_train.shape}, Validation labels size: {y_valid.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

print("Feature scaling complete.")

param_dist = {
    'n_estimators': [100, 200, 300, 400],        
    'max_features': ['sqrt', 'log2'],             
    'max_depth': [10, 20, 30, 40, 50, None],      
    'min_samples_split': [2, 5, 10],              
    'min_samples_leaf': [1, 2, 4]                 
}

rf_model = RandomForestClassifier(random_state=42)

#Random Search with 5-fold cross-validation
random_search = RandomizedSearchCV(estimator=rf_model, 
                                   param_distributions=param_dist, 
                                   n_iter=20, 
                                   cv=5, 
                                   verbose=2, 
                                   random_state=42, 
                                   n_jobs=-1)

print("Starting hyperparameter tuning with Randomized Search")
random_search.fit(X_train_scaled, y_train)
print("Tuning complete")

#best combination of hyperparameters found
print("\nBest hyperparameters found:")
print(random_search.best_params_)

best_model = random_search.best_estimator_
y_pred_tuned = best_model.predict(X_valid_scaled)
tuned_accuracy = accuracy_score(y_valid, y_pred_tuned)
print(f"\nAccuracy of the tuned Random Forest model: {tuned_accuracy * 100:.2f}%")


# Load the final test dataset
df_test = pd.read_csv('data/test.csv')
X_final_test = df_test[features_to_use]

X_final_test_imputed = imputer.transform(X_final_test)
X_final_test_scaled = scaler.transform(X_final_test_imputed)

final_predictions = best_model.predict(X_final_test_scaled)
print("Predictions for the final test set are complete!")

# Dictionary to map genre numbers to names
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

results_df = pd.DataFrame({
    'Artist Name': df_test['Artist Name'],
    'Track Name': df_test['Track Name'],
    'Predicted Genre': final_predictions
})
results_df['Predicted Genre Name'] = results_df['Predicted Genre'].map(genre_mapping)

# Saving the results to a CSV file
results_df.to_csv('results/predictions_RandomizedSearchCV.csv', index=False)

print("\nResults file 'predictions_RandomizedSearchCV.csv' has been created.")
print(results_df.head())

first_song_artist = df_test.iloc[0]['Artist Name']
first_song_track = df_test.iloc[0]['Track Name']
predicted_genre = results_df.iloc[0]['Predicted Genre']

print("\nAI Music Critic Review:")
print(f"Track: '{first_song_track}' by {first_song_artist}")
print(f"After analyzing its audio features, my model classifies this track as genre number {predicted_genre}.")