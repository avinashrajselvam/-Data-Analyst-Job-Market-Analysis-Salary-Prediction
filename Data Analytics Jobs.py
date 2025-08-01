import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Step 1: Load Dataset
data = pd.read_csv('Cleaned_DataAnalyst.csv')

# Step 2: Clean Salary Column
data = data[data['Salary Estimate'].notna()]
data['Salary Estimate'] = data['Salary Estimate'].str.replace(r'[^0-9\-]', '', regex=True)
data['Min Salary'] = data['Salary Estimate'].str.extract(r'(\d+)').astype(float)
data['Max Salary'] = data['Salary Estimate'].str.extract(r'-\s*(\d+)').astype(float)
data['Avg Salary'] = (data['Min Salary'] + data['Max Salary']) / 2
data.drop('Salary Estimate', axis=1, inplace=True)

# Step 3: Fill Missing Values
if 'Rating' in data.columns:
    data['Rating'].fillna(data['Rating'].median(), inplace=True)

threshold = len(data) * 0.3
data = data.dropna(thresh=threshold, axis=1)

categorical_cols = ['Company Name', 'Industry', 'Sector', 'Type of ownership']
for col in categorical_cols:
    if col in data.columns:
        data[col] = data[col].fillna(method='ffill')

# Step 4: Feature Engineering
data['Python'] = data['Job Description'].str.contains('Python', case=False, na=False).astype(int)
data['Excel'] = data['Job Description'].str.contains('Excel', case=False, na=False).astype(int)
data['Tech_Skills'] = data['Python'] + data['Excel']

data[['City', 'State']] = data['Location'].str.split(',', n=1, expand=True)

data['Founded'] = pd.to_numeric(data['Founded'], errors='coerce')
data['Founded'].fillna(data['Founded'].median(), inplace=True)

# Step 5: Drop rows with missing target/features
features = ['Rating', 'Tech_Skills', 'Founded']
data = data.dropna(subset=features + ['Avg Salary'])

# Step 6: Visualizations

# 1. Correlation Matrix
numeric_data = data.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.savefig('correlation_matrix.png')
plt.close()

# 2. Salary Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['Avg Salary'], kde=True, bins=20, color='skyblue')
plt.title("Average Salary Distribution")
plt.xlabel("Average Salary")
plt.savefig('salary_distribution.png')
plt.close()

# 3. Ratings by Industry
if 'Industry' in data.columns:
    plt.figure(figsize=(14, 6))
    sns.boxplot(x='Industry', y='Rating', data=data)
    plt.xticks(rotation=90)
    plt.title("Company Ratings by Industry")
    plt.tight_layout()
    plt.savefig('ratings_by_industry.png')
    plt.close()

# 4. Top 10 Jobs
top_jobs = data['Job Title'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_jobs.values, y=top_jobs.index, palette='viridis')
plt.title("Top 10 Job Titles")
plt.xlabel("Number of Postings")
plt.savefig('top_10_jobs.png')
plt.close()

# 5. Average Salary by Job Title
top_salary_jobs = data.groupby('Job Title')['Avg Salary'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_salary_jobs.values, y=top_salary_jobs.index, palette='mako')
plt.title("Top 10 Average Salaries by Job Title")
plt.xlabel("Average Salary")
plt.savefig('avg_salary_by_job_title.png')
plt.close()

# 6. Top 20 Locations
top_locations = data['Location'].value_counts().head(20)
plt.figure(figsize=(12, 8))
sns.barplot(x=top_locations.values, y=top_locations.index, palette='coolwarm')
plt.title("Top 20 Job Locations")
plt.xlabel("Number of Postings")
plt.savefig('top_20_locations.png')
plt.close()

# 7. Size Distribution
if 'Size' in data.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(y='Size', data=data, order=data['Size'].value_counts().index, palette='Set3')
    plt.title("Distribution of Company Sizes")
    plt.savefig('company_size_distribution.png')
    plt.close()

# 8. Average Salary by Company Size
if 'Size' in data.columns:
    avg_salary_by_size = data.groupby('Size')['Avg Salary'].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=avg_salary_by_size.values, y=avg_salary_by_size.index, palette='rocket')
    plt.title("Average Salary by Company Size")
    plt.xlabel("Average Salary")
    plt.savefig('avg_salary_by_size.png')
    plt.close()

# 9. Top 20 Types of Ownership
if 'Type of ownership' in data.columns:
    top_ownerships = data['Type of ownership'].value_counts().head(20)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_ownerships.values, y=top_ownerships.index, palette='cubehelix')
    plt.title("Top 20 Types of Ownership")
    plt.savefig('top_20_ownership.png')
    plt.close()

# 10. Sector Distribution
if 'Sector' in data.columns:
    top_sectors = data['Sector'].value_counts().head(20)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_sectors.values, y=top_sectors.index, palette='icefire')
    plt.title("Top 20 Sectors")
    plt.savefig('top_20_sectors.png')
    plt.close()

# 11. Average Salary by Sector
if 'Sector' in data.columns:
    avg_salary_by_sector = data.groupby('Sector')['Avg Salary'].mean().sort_values(ascending=False).head(20)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=avg_salary_by_sector.values, y=avg_salary_by_sector.index, palette='Blues_d')
    plt.title("Average Salary by Sector")
    plt.xlabel("Average Salary")
    plt.savefig('avg_salary_by_sector.png')
    plt.close()

# Step 7: Model Training
X = data[features]
y = data['Avg Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nModel Performance:\nMean Absolute Error (MAE): {mae:.2f}\nR² Score: {r2:.2f}")

# Step 8: Top Jobs by Salary and Rating
if 'Job Title' in data.columns and 'Company Name' in data.columns:
    best_jobs = data[['Job Title', 'Company Name', 'Avg Salary', 'Rating']].sort_values(
        by=['Avg Salary', 'Rating'], ascending=False)
    print("\nTop 10 Jobs by Salary and Rating:")
    print(best_jobs.head(10))
