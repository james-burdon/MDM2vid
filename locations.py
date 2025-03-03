import pandas as pd
import plotly.graph_objects as go
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset (adjust the path accordingly)
df = pd.read_csv("C:\\Users\\jburd\\Desktop\\MDM2vid\\combined_jobs_dataset.csv")

# Ensure 'Salary_Avg' is numeric and clean up any missing values
df["Salary_Avg"] = pd.to_numeric(df["Salary_Avg"], errors="coerce")
df = df.dropna(subset=["Salary_Avg", "State", "Job Description"])

# Text Preprocessing
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Clean text data by lemmatizing and removing stopwords."""
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)  # Keep only letters and spaces
    words = text.split()  # Split into a list of words
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatize & remove stopwords
    return " ".join(words)

# Clean the "Job Description" column
df["Cleaned_Description"] = df["Job Description"].fillna("").apply(clean_text)

# **TF-IDF Vectorization**
vectorizer = TfidfVectorizer(stop_words="english", max_features=3000, min_df=5, ngram_range=(1, 2))
X = vectorizer.fit_transform(df["Cleaned_Description"])

# Get the top 10 most common skills for each state
df["Extracted_Skills"] = X.argmax(axis=1)  # Assign the most probable skill to each job

# Get the feature names (skills) from the TF-IDF vectorizer
features = vectorizer.get_feature_names_out()

# Map the indices to actual skill names
df["Extracted_Skills"] = df["Extracted_Skills"].apply(lambda x: features[x])

# Group by state and calculate the average salary for each state
state_salary_avg = df.groupby("State")["Salary_Avg"].mean().reset_index()

# Get the most common skills in each state
state_skills = df.groupby("State")["Extracted_Skills"].apply(lambda x: ', '.join(x.value_counts().index[:5])).reset_index()

# Merge the salary data with the skills data
state_salary_avg = state_salary_avg.merge(state_skills, on="State")

# Manually input latitude and longitude for each state (approximate centers)
state_coordinates = {
    'Alabama': {'lat': 32.8067, 'lon': -86.7911},
    'Alaska': {'lat': 61.3704, 'lon': -149.4937},
    'Arizona': {'lat': 33.7298, 'lon': -111.4312},
    'Arkansas': {'lat': 34.9697, 'lon': -92.3731},
    'California': {'lat': 36.7783, 'lon': -119.4179},
    'Colorado': {'lat': 39.5501, 'lon': -105.7821},
    'Connecticut': {'lat': 41.6032, 'lon': -73.0877},
    'Delaware': {'lat': 38.8026, 'lon': -75.5277},
    'Florida': {'lat': 27.9944, 'lon': -81.7603},
    'Georgia': {'lat': 33.7691, 'lon': -84.6701},
    'Hawaii': {'lat': 21.0943, 'lon': -157.4981},
    'Idaho': {'lat': 44.2998, 'lon': -114.7420},
    'Illinois': {'lat': 40.6331, 'lon': -89.3985},
    'Indiana': {'lat': 39.7910, 'lon': -86.1477},
    'Iowa': {'lat': 41.8780, 'lon': -93.0977},
    'Kansas': {'lat': 38.5266, 'lon': -96.7265},
    'Kentucky': {'lat': 37.6681, 'lon': -84.6701},
    'Louisiana': {'lat': 31.9686, 'lon': -99.9018},
    'Maine': {'lat': 44.2998, 'lon': -69.4455},
    'Maryland': {'lat': 39.0458, 'lon': -76.6413},
    'Massachusetts': {'lat': 42.4072, 'lon': -71.3824},
    'Michigan': {'lat': 44.3148, 'lon': -85.6024},
    'Minnesota': {'lat': 45.6945, 'lon': -93.9002},
    'Mississippi': {'lat': 32.7416, 'lon': -89.6682},
    'Missouri': {'lat': 38.5739, 'lon': -92.6038},
    'Montana': {'lat': 46.8797, 'lon': -110.3626},
    'Nebraska': {'lat': 41.1254, 'lon': -98.2681},
    'Nevada': {'lat': 38.3135, 'lon': -117.0554},
    'New Hampshire': {'lat': 33.7490, 'lon': -84.3880},
    'New Jersey': {'lat': 40.0583, 'lon': -74.4057},
    'New Mexico': {'lat': 34.5194, 'lon': -105.8701},
    'New York': {'lat': 40.7128, 'lon': -74.0060},
    'North Carolina': {'lat': 35.6301, 'lon': -79.8060},
    'North Dakota': {'lat': 47.5515, 'lon': -101.0020},
    'Ohio': {'lat': 40.4173, 'lon': -82.9071},
    'Oklahoma': {'lat': 35.4676, 'lon': -97.5164},
    'Oregon': {'lat': 43.8041, 'lon': -120.5542},
    'Pennsylvania': {'lat': 40.2737, 'lon': -76.8844},
    'Rhode Island': {'lat': 41.6809, 'lon': -71.5118},
    'South Carolina': {'lat': 33.8361, 'lon': -81.1637},
    'South Dakota': {'lat': 44.2998, 'lon': -99.4388},
    'Tennessee': {'lat': 35.5175, 'lon': -86.5804},
    'Texas': {'lat': 31.9686, 'lon': -99.9018},
    'Utah': {'lat': 40.7608, 'lon': -111.8624},
    'Vermont': {'lat': 44.0682, 'lon': -72.6023},
    'Virginia': {'lat': 37.4316, 'lon': -78.6569},
    'Washington': {'lat': 47.7511, 'lon': -120.7401},
    'West Virginia': {'lat': 38.5976, 'lon': -80.4549},
    'Wisconsin': {'lat': 43.7844, 'lon': -88.7879},
    'Wyoming': {'lat': 43.0759, 'lon': -107.2903},
}

# Add the latitude and longitude to the DataFrame
state_salary_avg['Latitude'] = state_salary_avg['State'].map(lambda x: state_coordinates.get(x, {}).get('lat', None))
state_salary_avg['Longitude'] = state_salary_avg['State'].map(lambda x: state_coordinates.get(x, {}).get('lon', None))

# Create the map using Plotly's graph_objects
fig = go.Figure()

# Add the choropleth map for average salary
fig.add_trace(go.Choropleth(
    z=state_salary_avg['Salary_Avg'],
    locations=state_salary_avg['State'],
    locationmode="USA-states",
    colorscale="Viridis",
    colorbar_title="Average Salary",
    hovertemplate='<b>%{location}</b><br>Salary: %{z}<br>Skills: %{customdata[0]}',
    customdata=state_salary_avg[['Extracted_Skills']].values
))

# Add text annotations for state names
fig.add_trace(go.Scattergeo(
    locations=state_salary_avg['State'],
    locationmode="USA-states",
    text=state_salary_avg['State'],
    mode='text',
    textfont=dict(size=10, color='black'),
    showlegend=False
))

# Update geos for the map
fig.update_geos(
    scope="usa",
    projection_type="albers usa",  # Use the Albers projection for USA
    showlakes=True,
    lakecolor='rgb(255, 255, 255)',
    center=dict(lon=-98, lat=38),  # Center the map for better visibility
    projection_scale=1,  # Adjust zoom level for better fit
)

fig.update_layout(
    title_text='Average Salary by State with Skills in the US',
    geo=dict(
        lakecolor='rgb(255, 255, 255)',
        projection_scale=1  # Adjust zoom level for better fit
    )
)

fig.show()
