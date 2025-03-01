import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# custom stopwords that wont be included
custom_stopwords = {
    'data', 'analysis', 'analyst', 'analytic', 'analytics', 'science', 'scientist', 'role', 'skill',
    'skills', 'tool', 'tools', 'model', 'models', 'algorithm', 'algorithms',
    'process', 'project', 'task', 'knowledge', 'ability', 'team', 'work',
    'experience', 'job', 'position', 'responsibilities', 'requirements',
    'qualifications', 'including', 'develop', 'developing', 'implement',
    'implementing', 'using', 'utilize', 'utilizing', 'support', 'collaborate',
    'collaborating', 'perform', 'performing', 'strong', 'proficient',
    'understanding', 'demonstrated', 'excellent', 'required', 'preferred',
    'opportunity', 'environment', 'company', 'organization', 'business','statistics','statistical','client','report','reporting','management','information','employee','product','customer','service'
}

# function to process the text data
def preprocess_text(text):
    # lowercase
    text = text.lower()
    # remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # tokenise
    tokens = word_tokenize(text)
    # remove stopwords (default + custom)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word not in custom_stopwords]
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# load data
df = pd.read_csv('combined_jobs_dataset.csv')

#process
df['Job Description'] = df['Job Description'].apply(preprocess_text)
 
# automatically create 3 salary bands
df['Salary_Band'] = pd.qcut(df['Salary_Avg'], q=3, labels=['Low', 'Medium', 'High'])

# Get the min and max salary for each band
salary_ranges = {}
for band in ['Low', 'Medium', 'High']:
    band_salaries = df[df['Salary_Band'] == band]['Salary_Avg']
    salary_ranges[band] = (band_salaries.min(), band_salaries.max())

# Print the unique salary bands with their ranges
print("Salary Bands with Ranges:")
for band, (min_sal, max_sal) in salary_ranges.items():
    print(f"{band}: ${min_sal:,.2f} - ${max_sal:,.2f}")

#group by salary band
salary_bands = {
    'Low': df[df['Salary_Band'] == 'Low']['Job Description'],
    'Medium': df[df['Salary_Band'] == 'Medium']['Job Description'],
    'High': df[df['Salary_Band'] == 'High']['Job Description']
}

#vector the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')

# apply NMF for each salary band
num_topics = 5
nmf_model = NMF(n_components=num_topics, random_state=42)
 
for band, text in salary_bands.items():
    tfidf = tfidf_vectorizer.fit_transform(text)
    nmf = nmf_model.fit(tfidf)
    
    #get the top words for each topic
    feature_names = tfidf_vectorizer.get_feature_names_out()
    top_words_per_topic = {}
    for topic_idx, topic in enumerate(nmf.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]  # Top 10 words per topic
        top_words_per_topic[f"Topic {topic_idx}"] = top_words
    
    # combine top words into a single string for word cloud
    all_top_words = ' '.join([' '.join(words) for words in top_words_per_topic.values()])
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_top_words)
    
    min_sal, max_sal = salary_ranges[band]
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f"Word Cloud for {band} Salary Band (${min_sal:,.2f} - ${max_sal:,.2f})", fontsize=15)
    plt.axis('off')
    plt.show()