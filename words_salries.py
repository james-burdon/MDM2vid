import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords               
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from wordcloud import WordCloud
from scipy import stats
from collections import Counter
import random
from tqdm import tqdm  #

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
    'opportunity', 'environment', 'company', 'organization', 'business','statistics','statistical','client','report','reporting','management','information','employee','product','customer','service','spark','de','scientific'
}
stop_words = set(stopwords.words('english')).union(custom_stopwords)

# function to process the text data
def preprocess_text(text):
    # lowercase
    text = text.lower()
    # remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # tokenise
    tokens = word_tokenize(text)
    # remove stopwords (default + custom)
    tokens = [word for word in tokens if word not in stop_words]
    
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
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=list(stop_words))

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

#hypothesis testing

# Function to extract word frequency counts from each salary band
def get_word_frequencies(text_series):
    # Combine all text in the series
    combined_text = ' '.join(text_series)
    # Split into words
    words = combined_text.split()
    # Count word frequencies
    return Counter(words)

# Get word frequencies for each salary band
low_freq = get_word_frequencies(salary_bands['Low'])
medium_freq = get_word_frequencies(salary_bands['Medium'])
high_freq = get_word_frequencies(salary_bands['High'])

# Get unique words across all bands
all_words = set()
for counter in [low_freq, medium_freq, high_freq]:
    all_words.update(counter.keys())

# Find words that appear in at least two salary bands
common_words = []
for word in all_words:
    bands_with_word = 0
    if word in low_freq and low_freq[word] >= 5:
        bands_with_word += 1
    if word in medium_freq and medium_freq[word] >= 5:
        bands_with_word += 1
    if word in high_freq and high_freq[word] >= 5:
        bands_with_word += 1
    
    if bands_with_word >= 2:
        common_words.append(word)

# Create a DataFrame to store word frequencies across bands
word_freq_df = pd.DataFrame({
    'Word': common_words,
    'Low_Freq': [low_freq.get(word, 0) for word in common_words],
    'Medium_Freq': [medium_freq.get(word, 0) for word in common_words],
    'High_Freq': [high_freq.get(word, 0) for word in common_words]
})

# Normalize frequencies by total word count in each band
total_low = sum(low_freq.values())
total_medium = sum(medium_freq.values())
total_high = sum(high_freq.values())

word_freq_df['Low_Norm'] = word_freq_df['Low_Freq'] / total_low * 1000  # per 1000 words
word_freq_df['Medium_Norm'] = word_freq_df['Medium_Freq'] / total_medium * 1000
word_freq_df['High_Norm'] = word_freq_df['High_Freq'] / total_high * 1000

# Calculate the difference between high and low salary bands
word_freq_df['High_Low_Diff'] = word_freq_df['High_Norm'] - word_freq_df['Low_Norm']

# Sort by the absolute difference
word_freq_df['Abs_Diff'] = np.abs(word_freq_df['High_Low_Diff'])
word_freq_df = word_freq_df.sort_values('Abs_Diff', ascending=False)

# Chi-square test for most significant words
significant_words = []
p_values = []

# Take top 50 words with highest absolute difference for testing
for word in word_freq_df.head(50)['Word']:
    # Create a contingency table for this word
    observed = np.array([
        [low_freq.get(word, 0), total_low - low_freq.get(word, 0)],
        [medium_freq.get(word, 0), total_medium - medium_freq.get(word, 0)],
        [high_freq.get(word, 0), total_high - high_freq.get(word, 0)]
    ])
    
    # Perform chi-square test
    chi2, p, dof, expected = stats.chi2_contingency(observed)
    
    if p < 0.05:  # Using 0.05 significance level
        significant_words.append(word)
        p_values.append(p)

# Create a DataFrame for significant words
sig_words_df = pd.DataFrame({
    'Word': significant_words,
    'p_value': p_values,
    'Low_Freq_Norm': [word_freq_df[word_freq_df['Word'] == word]['Low_Norm'].values[0] for word in significant_words],
    'Medium_Freq_Norm': [word_freq_df[word_freq_df['Word'] == word]['Medium_Norm'].values[0] for word in significant_words],
    'High_Freq_Norm': [word_freq_df[word_freq_df['Word'] == word]['High_Norm'].values[0] for word in significant_words]
})

# Sort by p-value
sig_words_df = sig_words_df.sort_values('p_value')

# Print the most statistically significant words
print(f"Total significant words found: {len(significant_words)}")
print("\nTop 20 statistically significant words across salary bands:")
print(sig_words_df.head(20))

# Visualize the significant differences
plt.figure(figsize=(12, 10))

# Select top 15 words for visualization
plot_words = sig_words_df.head(15)['Word'].tolist()
x = np.arange(len(plot_words))
width = 0.25

# Plot the bars
plt.bar(x - width, 
        [word_freq_df[word_freq_df['Word'] == word]['Low_Norm'].values[0] for word in plot_words], 
        width, label=f'Low (${salary_ranges["Low"][0]:,.0f}-${salary_ranges["Low"][1]:,.0f})')
        
plt.bar(x, 
        [word_freq_df[word_freq_df['Word'] == word]['Medium_Norm'].values[0] for word in plot_words], 
        width, label=f'Medium (${salary_ranges["Medium"][0]:,.0f}-${salary_ranges["Medium"][1]:,.0f})')
        
plt.bar(x + width, 
        [word_freq_df[word_freq_df['Word'] == word]['High_Norm'].values[0] for word in plot_words], 
        width, label=f'High (${salary_ranges["High"][0]:,.0f}-${salary_ranges["High"][1]:,.0f})')

plt.xlabel('Words')
plt.ylabel('Frequency per 1,000 words')
plt.title('Most Significant Words Across Salary Bands')
plt.xticks(x, plot_words, rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()

# Calculate global statistics for standard chi-square test
print("\nStandard Global Chi-Square Test across all significant words:")
# Create a global contingency table
global_observed = np.zeros((3, 2))

for word in significant_words:
    global_observed[0, 0] += low_freq.get(word, 0)
    global_observed[1, 0] += medium_freq.get(word, 0)
    global_observed[2, 0] += high_freq.get(word, 0)

global_observed[0, 1] = total_low - global_observed[0, 0]
global_observed[1, 1] = total_medium - global_observed[1, 0]
global_observed[2, 1] = total_high - global_observed[2, 0]

chi2, p, dof, expected = stats.chi2_contingency(global_observed)
print(f"Chi-Square value: {chi2:.2f}")
print(f"p-value: {p:.10f}")
print(f"Degrees of freedom: {dof}")
if p < 0.05:
    print("Conclusion from standard test: The frequency distribution of significant words differs significantly across salary bands.")
else:
    print("Conclusion from standard test: No significant global difference in word frequencies across salary bands.")