import pandas as pd
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords

# Define custom stopwords
custom_stopwords = [
    'data', 'analysis', 'analyst', 'analytic', 'analytics', 'science', 'scientist', 'role', 'skill',
    'skills', 'tool', 'tools', 'model', 'models', 'algorithm', 'algorithms',
    'process', 'project', 'task', 'knowledge', 'ability', 'team', 'work',
    'experience', 'job', 'position', 'responsibilities', 'requirements',
    'qualifications', 'including', 'develop', 'developing', 'implement',
    'implementing', 'using', 'utilize', 'utilizing', 'support', 'collaborate',
    'collaborating', 'perform', 'performing', 'strong', 'proficient',
    'understanding', 'demonstrated', 'excellent', 'required', 'preferred',
    'opportunity', 'environment', 'company', 'organization', 'business', 'statistics', 'statistical',
    'client', 'report', 'reporting', 'management', 'information', 'employee', 'product', 'customer',
    'service', 'spark', 'de', 'scientific', 'insight', 'andor', 'may', 'related', 'year', 'status',
    'solution', 'database', 'quality', 'must', 'technology', 'etc', 'language', 'engineer', 'years','working','systems','solutions','technical', 'new',  'health', 'reports','provide','us','insights','analytical','software','degree'
]

# Combine custom stopwords with NLTK's English stopwords
# Convert set to list since TfidfVectorizer expects a list
stop_words = list(set(stopwords.words('english')).union(set(custom_stopwords)))

# Load the dataset (adjust the path accordingly)
df = pd.read_csv("combined_jobs_dataset.csv")

# Ensure 'Salary_Avg' is numeric and clean up any missing values
df["Salary_Avg"] = pd.to_numeric(df["Salary_Avg"], errors="coerce")
df = df.dropna(subset=["Salary_Avg", "State", "Job Description"])

# Text Preprocessing
def clean_text(text):
    """Clean text data by lemmatizing and removing stopwords."""
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)  # Keep only letters and spaces
    words = text.split()  # Split into a list of words
    return " ".join(words)

# Clean the "Job Description" column
df["Cleaned_Description"] = df["Job Description"].fillna("").apply(clean_text)

# **TF-IDF Vectorization to Extract Top 5 Skills**
# Use our combined stop_words, ensuring it's a list
vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=3000, min_df=5, ngram_range=(1, 2))
X = vectorizer.fit_transform(df["Cleaned_Description"])

# Get the feature names (skills) from the TF-IDF vectorizer
features = vectorizer.get_feature_names_out()

# Extract top 5 skills based on their frequency (TF-IDF scores)
top_skills = pd.DataFrame(X.toarray(), columns=features)
top_skills_mean = top_skills.mean(axis=0).sort_values(ascending=False).head(5)

# Get the top skills
top_5_skills = top_skills_mean.index.tolist()

# Region mapping for states (using initials)
state_to_region = {
    'NY': 'Northeast', 'VA': 'South', 'MA': 'Northeast', 'CA': 'West', 'IL': 'Midwest', 'MO': 'Midwest', 'WA': 'West', 
    'DC': 'Northeast', 'TN': 'South', 'TX': 'South', 'PA': 'Northeast', 'AZ': 'West', 'WI': 'Midwest', 'GA': 'South', 
    'FL': 'South', 'NE': 'Midwest', 'KS': 'Midwest', 'NH': 'Northeast', 'NJ': 'Northeast', 'LA': 'South', 'OH': 'Midwest', 
    'IN': 'Midwest', 'MD': 'South', 'CO': 'West', 'UT': 'West', 'OR': 'West', 'MI': 'Midwest', 'SC': 'South', 'MS': 'South', 
    'AL': 'South', 'RI': 'Northeast', 'IA': 'Midwest', 'MN': 'Midwest', 'OK': 'South', 'CT': 'Northeast', 'NC': 'South', 
    'DE': 'Northeast', 'WV': 'South'
}

# Map states to regions (using state initials)
df['Region'] = df['State'].map(state_to_region)

# **Check for Missing Regions**
missing_states = df[df['Region'].isna()]['State'].unique()
if len(missing_states) > 0:
    print("Warning: Missing regions for the following states:", missing_states)

# Now, calculate the region-wise average salary for each of these top skills
# Aggregate salary by skill and region
df["Skills"] = df["Cleaned_Description"].apply(lambda x: [skill for skill in top_5_skills if skill in x])

# We will consider regions and calculate the average salary based on the top skills
skills_by_region = []

for skill in top_5_skills:
    region_salary = df[df["Skills"].apply(lambda x: skill in x)].groupby("Region")["Salary_Avg"].mean().reset_index()
    region_salary["Skill"] = skill
    skills_by_region.append(region_salary)

# Combine all skill-based region salary data into a single DataFrame
skills_region_salary_df = pd.concat(skills_by_region)

# Check if the final DataFrame contains data
if skills_region_salary_df.empty:
    print("Warning: No data found for the top skills in regions.")
else:
    print("Data is ready for visualization.")

# Now we will visualize this data using Plotly
fig = go.Figure()

# Add a bar plot for each skill to visualize the best regions for that skill
for skill in top_5_skills:
    skill_data = skills_region_salary_df[skills_region_salary_df["Skill"] == skill]
    
    fig.add_trace(go.Bar(
        x=skill_data["Region"],
        y=skill_data["Salary_Avg"],
        name=skill,
        hovertemplate='<b>%{x}</b><br>Average Salary: %{y}<br>Skill: ' + skill,
        marker=dict(line=dict(color='rgb(8,48,107)', width=1.5))
    ))

# Update layout and aesthetics
fig.update_layout(
    title="Best Regions for Top Skills Based on Average Salary",
    xaxis_title="Region",
    yaxis_title="Average Salary",
    barmode="group",  # Group bars by skill
    showlegend=True,
)

fig.show()