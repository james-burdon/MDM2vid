import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Load data
file_path = "Cleaned_DS_Jobs.csv"
df = pd.read_csv(file_path)

# Extract city and state information
df['City'] = df['Location'].apply(lambda x: x.split(',')[0])  # Extract city
df['State'] = df['job_state']  # State information provided

# Encode city and state
le_city = LabelEncoder()
le_state = LabelEncoder()
df['City_encoded'] = le_city.fit_transform(df['City'])
df['State_encoded'] = le_state.fit_transform(df['State'])

# Perform location-based clustering
location_features = df[['City_encoded', 'State_encoded']]
kmeans_location = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Location_Cluster'] = kmeans_location.fit_predict(location_features)

# Display sample cities for each cluster
grouped_locations = df[['City', 'State', 'Location_Cluster']].groupby('Location_Cluster').head(3)
print("Location clustering examples:")
for cluster in range(5):
    print(f"\nCluster {cluster}:")
    print(grouped_locations[grouped_locations['Location_Cluster'] == cluster][['City', 'State']])

# Perform job-based clustering
job_texts = df['Job Title'] + " " + df['Job Description']
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
tfidf_matrix = tfidf_vectorizer.fit_transform(job_texts)

# Apply SVD for dimensionality reduction
svd = TruncatedSVD(n_components=50, random_state=42)
reduced_tfidf_matrix = svd.fit_transform(tfidf_matrix)

# Perform KMeans clustering
kmeans_jobs = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Job_Cluster'] = kmeans_jobs.fit_predict(reduced_tfidf_matrix)

# Save clustered data
df.to_csv("Clustered_DS_Jobs.csv", index=False)

# Display first few rows
print(df[['Company Name', 'Location', 'State', 'Location_Cluster', 'Job Title', 'Job_Cluster']].head())
