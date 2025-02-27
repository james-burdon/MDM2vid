# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack

# Load the CSV file
file_path = "Cleaned_DS_Jobs.csv"  # Replace with your actual file path
df = pd.read_csv(file_path)

# Select the "Job Description" column and remove missing values
job_descriptions = df["Job Description"].dropna().head(500)  # Process only the first 500 rows for efficiency

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=20)  # Limit to extracting 20 keywords

# Train the TF-IDF model and transform the text data
tfidf_matrix = tfidf_vectorizer.fit_transform(job_descriptions)

# Get the extracted keywords
feature_names = tfidf_vectorizer.get_feature_names_out()

# Compute the importance score for each word (take the mean TF-IDF score)
mean_tfidf_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()

# Sort keywords by importance score in descending order
keywords = sorted(zip(feature_names, mean_tfidf_scores), key=lambda x: x[1], reverse=True)

# Display the top 10 highest-ranked keywords
for word, score in keywords[:10]:
    print(f"{word}: {score:.4f}")

# Load the dataset
df = pd.read_csv("Cleaned_DS_Jobs.csv")

# Remove spaces and split salary into min and max values
df["Salary Estimate"] = df["Salary Estimate"].str.strip()  # Remove extra spaces
df[["Min Salary", "Max Salary"]] = df["Salary Estimate"].str.split("-", expand=True)

# Convert to numeric values
df["Min Salary"] = pd.to_numeric(df["Min Salary"], errors="coerce")
df["Max Salary"] = pd.to_numeric(df["Max Salary"], errors="coerce")

# Compute the average salary
df["Avg Salary"] = (df["Min Salary"] + df["Max Salary"]) / 2

# Categorize into salary groups
df["Salary Category"] = pd.cut(df["Avg Salary"], bins=[0, 100, 150, 300], labels=["Low", "Medium", "High"])

# Display distribution of salary categories
print(df["Salary Category"].value_counts())

# Data preprocessing
df["Avg Salary"] = (df["Min Salary"] + df["Max Salary"]) / 2

# Select the features that need to be clustered
text_features = df[["Job Title", "Company Name", "Location"]].fillna("")

# Processing text data
vectorizer = TfidfVectorizer(stop_words="english", max_features=100)  # Take the first 100 keywords
job_title_tfidf = vectorizer.fit_transform(text_features["Job Title"])
company_tfidf = vectorizer.fit_transform(text_features["Company Name"])
location_tfidf = vectorizer.fit_transform(text_features["Location"])

# Standardized salary data
scaler = StandardScaler()
salary_scaled = scaler.fit_transform(df[["Avg Salary"]].fillna(df["Avg Salary"].median()))

# Combine all features
X = hstack([job_title_tfidf, company_tfidf, location_tfidf, salary_scaled])

# Determine the best K value
wcss = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Drawing the image
plt.figure(figsize=(8,5))
plt.plot(range(2, 10), wcss, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
plt.title("Elbow Method for Optimal K")
plt.show()

# Perform clustering(Suggest K=4)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X)

# Visualize the salary levels of different clusters
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["Cluster"], y=df["Avg Salary"])
plt.xlabel("Cluster")
plt.ylabel("Average Salary")
plt.title("Salary Distribution by Cluster")
plt.show()

# View the job distribution for each cluster
print(df.groupby("Cluster")["Job Title"].value_counts().head(10))