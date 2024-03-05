
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sqlalchemy import create_engine
# Load your dataset
users_df1= pd.read_csv("users.csv")  # Replace "your_dataset.csv" with the path to your dataset
# Assuming you have loaded your dataset into df1

# Define the categories for frontend and backend developers
frontend_languages = ['Python', 'C++', 'Java', 'JavaScript', 'Django', 'HTML', 'C', 'Reactjs', 'CSS']
backend_languages = ['MySQL', 'NoSQL', 'SQL', 'PHP']

# Create a new binary target variable: 1 for frontend developer, 0 for backend developer
users_df1['Developer Type'] = users_df1['Primary Language'].apply(lambda x: 1 if x in frontend_languages else (0 if x in backend_languages else None))

# Remove rows with missing values
users_df1.dropna(subset=['Developer Type'], inplace=True)

# Select features and target variable
X = users_df1[['Badge Value', 'Reputation', 'Acceptance Rate', 'Followers', 'Forks', 'Stars', 'Commits']]
y = users_df1['Developer Type']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Initialize and train SVM model
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Predict with SVM model
svm_pred = svm_model.predict(X_test)

# Filter data based on predictions
predictions = users_df1.loc[X_test.index[svm_pred == 1]]
#predictions
frontend_df = predictions[predictions['Developer Type'] == 1]
backend_df = predictions[predictions['Developer Type'] == 0]
frontend_df
frontend_df
# Save frontend_df to CSV
frontend_df.to_csv('frontend_developers.csv', index=False)

# Save backend_df to CSV
backend_df.to_csv('backend_developers.csv', index=False)
engine = create_engine('postgresql://postgres:1234@localhost:5432/postgres')
frontend_df.to_sql('frontend_developers', engine, if_exists='replace', index=False)

# Store backend_df into PostgreSQL
backend_df.to_sql('backend_developers', engine, if_exists='replace', index=False)
