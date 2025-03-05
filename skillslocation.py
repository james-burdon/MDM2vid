import pandas as pd
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Load the dataset (adjust the path accordingly)
df = pd.read_csv("C:\\Users\\jburd\\Desktop\\MDM2vid\\combined_jobs_dataset.csv")

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
vectorizer = TfidfVectorizer(stop_words="english", max_features=3000, min_df=5, ngram_range=(1, 2))
X = vectorizer.fit_transform(df["Cleaned_Description"])

# Get the feature names (skills) from the TF-IDF vectorizer
features = vectorizer.get_feature_names_out()

# Extract top 5 skills based on their frequency (TF-IDF scores)
top_skills = pd.DataFrame(X.toarray(), columns=features)
top_skills_mean = top_skills.mean(axis=0).sort_values(ascending=False).head(5)

# Get the top skills
top_5_skills = top_skills_mean.index.tolist()

# Now, calculate the state-wise average salary for each of these top skills
# Aggregate salary by skill and state
df["Skills"] = df["Cleaned_Description"].apply(lambda x: [skill for skill in top_5_skills if skill in x])

# We will consider states and calculate the average salary based on the top skills
skills_by_state = []

for skill in top_5_skills:
    state_salary = df[df["Skills"].apply(lambda x: skill in x)].groupby("State")["Salary_Avg"].mean().reset_index()
    state_salary["Skill"] = skill
    skills_by_state.append(state_salary)

# Combine all skill-based state salary data into a single DataFrame
skills_state_salary_df = pd.concat(skills_by_state)

# Now we will visualize this data using Plotly

fig = go.Figure()

# Add a scatter plot for each skill to visualize the best states for that skill
for skill in top_5_skills:
    skill_data = skills_state_salary_df[skills_state_salary_df["Skill"] == skill]
    
    fig.add_trace(go.Bar(
        x=skill_data["State"],
        y=skill_data["Salary_Avg"],
        name=skill,
        hovertemplate='<b>%{x}</b><br>Average Salary: %{y}<br>Skill: ' + skill,
        marker=dict(line=dict(color='rgb(8,48,107)', width=1.5))
    ))

# Update layout and aesthetics
fig.update_layout(
    title="Best States for Top Skills Based on Average Salary",
    xaxis_title="State",
    yaxis_title="Average Salary",
    barmode="group",  # Group bars by skill
    xaxis=dict(tickangle=45),
    showlegend=True,
)

fig.show()
