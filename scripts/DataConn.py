from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
import psycopg2

# Load SQL magic commands
%load_ext sql

# Connect to PostgreSQL database
%sql postgresql://postgres:1234@localhost:5432/postgres

# Create SQLAlchemy engine
engine = create_engine('postgresql://postgres:1234@localhost:5432/postgres')

# Read data from CSV file
data = pd.read_csv("finaldata.csv")

# Write data to PostgreSQL table
data.to_sql('gitstack', engine, if_exists='replace', index=False)

# Execute SQL query to select data from PostgreSQL table
result = %sql SELECT * FROM gitstack

# Convert SQL result to DataFrame
df1 = result.DataFrame()

# Feature selection
X = df1[['Badge Value', 'Reputation', 'Acceptance Rate', 'Followers', 'Forks', 'Stars', 'Commits']]

# Target variable
df1['User Type'] = pd.cut(df1['Acceptance Rate'], bins=[1, 50, 75, 100], labels=[0, 1, 2])
y = df1['User Type']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Initialize and train the Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Display classification report
print(classification_report(y_test, y_pred))

# Read test data from CSV file
test_data = pd.read_csv("test.csv")

# Make predictions on the test data
X_test = test_data[['Badge Value', 'Reputation', 'Acceptance Rate', 'Followers', 'Forks', 'Stars', 'Commits']]
predictions = model.predict(X_test)

# Add predictions to test data
test_data['Predictions'] = predictions

# Save test data with predictions to CSV file
test_data.to_csv("test_predictions.csv", index=False)
