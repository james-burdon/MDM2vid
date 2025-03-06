import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import os

# Define a custom set of stopwords to remove common but uninformative words
custom_stopwords = set(["and", "to", "of", "the", "in", "for", "with", "on", "at", "by", "from", "is", "this", "that", "as", "it", "be", "or", "are", "an", "was", "were", "has", "have", "had", "a", "we", "can", "not", "will", "your", "our", "their", "they", "you", "about", "more", "other"])

# Load the updated dataset / 读取更新后的数据集
csv_file_path = "combined_jobs_dataset.csv"
df = pd.read_csv(csv_file_path)

# Ensure output directory exists / 确保输出目录存在
output_dir = "output_plots"
os.makedirs(output_dir, exist_ok=True)

# Function to clean text / 清理文本函数
def clean_text(text):
    text = str(text).lower()  # Convert to lowercase / 转换为小写
    text = re.sub(r'\d+', '', text)  # Remove numbers / 去除数字
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation / 去除标点符号
    text = text.strip()  # Trim whitespace / 去除前后空格
    return text

# Process Job Title Column / 处理 Job Title 字段
df['cleaned_job_title'] = df['Job Title'].apply(clean_text)

# Count Most Frequent Words / 统计最常见的词
all_words = ' '.join(df['cleaned_job_title']).split()
word_freq = Counter(all_words)

# Select Top 20 Most Frequent Words / 选取前 20 个最高频词汇
top_keywords = word_freq.most_common(20)
print("Top 20 Keywords:")
for word, count in top_keywords:
    print(f"{word}: {count} times")

# Define Clustering Rules
# Clustering logic：
# 1. Data Engineer: Include both ‘data’ and ‘engineer’
# 2. Data Scientist: contains only ‘data’
# 3. Software/ML Engineer: contains only ‘engineer’
# 4. Other: does not contain ‘data’ or ‘engineer’
def categorize_job(title):
    title_lower = title.lower()
    contains_data = "data" in title_lower
    contains_engineer = "engineer" in title_lower

    if contains_data and contains_engineer:
        return "Data Engineer"
    elif contains_data:
        return "Data Scientist"
    elif contains_engineer:
        return "Software/ML Engineer"
    else:
        return "Other"

# Apply Classification Rules / 应用分类规则
df['Job Category'] = df['cleaned_job_title'].apply(categorize_job)

# Count Job Categories / 统计各类别的职位数量
job_category_counts = df['Job Category'].value_counts()

# Use 'Salary_Avg' instead of 'Salary Estimate' / 使用 'Salary_Avg' 代替 'Salary Estimate'
df['Salary (Processed)'] = df['Salary_Avg']

# Compute Average Salary for Each Category / 计算每个类别的平均薪资
salary_by_category = df.groupby('Job Category', sort=False)['Salary (Processed)'].mean().sort_values(ascending=False)

# Extract top 5 most frequent meaningful words from job descriptions per cluster / 提取每个聚类的职位描述中最有意义的五个词
def extract_top_words(description_series, top_n=5):
    text = ' '.join(description_series.dropna()).lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = text.split()
    words = [word for word in words if word not in custom_stopwords]  # Remove custom stopwords
    word_counts = Counter(words)
    return word_counts.most_common(top_n)

cluster_top_words = {}
for category in df['Job Category'].unique():
    cluster_top_words[category] = extract_top_words(df[df['Job Category'] == category]['Job Description'])

# Print the top words per cluster
for category, words in cluster_top_words.items():
    print(f"Top words for {category}:")
    for word, count in words:
        print(f"{word}: {count} times")
    print("\n")

# Display Top 20 Keywords / 显示前 20 个最高频词汇
top_keywords_df = pd.DataFrame(top_keywords, columns=['Keyword', 'Count'])

# Plot Job Category Distribution / 绘制职位类别分布柱状图
plt.figure(figsize=(12, 6))
sns.barplot(x=job_category_counts.index, y=job_category_counts.values, hue=job_category_counts.index, palette='Blues', legend=False)
plt.xlabel('Job Category')
plt.ylabel('Number of Jobs')
plt.title('Job Distribution by Category')
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "job_category_distribution.png"), dpi=300, bbox_inches='tight')  # Save the plot
plt.show()

# Plot Average Salary by Job Category (sorted by salary) / 绘制按薪资排序的平均薪资柱状图
plt.figure(figsize=(12, 6))
sns.barplot(x=salary_by_category.index, y=salary_by_category.values, hue=salary_by_category.index, palette='viridis', order=salary_by_category.index, legend=False)
plt.xlabel('Job Category')
plt.ylabel('Average Salary ($1000s)')
plt.title('Average Salary by Job Category')
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "average_salary_by_category.png"), dpi=300, bbox_inches='tight')  # Save the plot
plt.show()
