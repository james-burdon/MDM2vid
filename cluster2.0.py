import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

# 读取 CSV 数据 / Load CSV Data
file_path = "Cleaned_DS_Jobs.csv"  # 请确保此文件在你的项目目录中
df = pd.read_csv(file_path)

# 清理文本函数 / Function to Clean Text
def clean_text(text):
    text = str(text).lower()  # 确保文本为字符串并转换为小写
    text = re.sub(r'\d+', '', text)  # 去除数字
    text = re.sub(r'[^\w\s]', '', text)  # 去除标点符号
    text = text.strip()  # 去除前后空格
    return text

# 处理 Job Title 字段 / Process Job Title Column
df['cleaned_job_title'] = df['Job Title'].apply(clean_text)

# 统计最常见的词 / Count Most Frequent Words
all_words = ' '.join(df['cleaned_job_title']).split()
word_freq = Counter(all_words)

# 选取前 20 个最高频词汇 / Select Top 20 Most Frequent Words
top_keywords = word_freq.most_common(20)
print("Top 20 Keywords:")
for word, count in top_keywords:
    print(f"{word}: {count} times")

# 创建分类规则 / Define Clustering Rules
# 分类逻辑：
# 1. Data Engineer: 同时包含 "data" 和 "engineer"
# 2. Data Scientist: 仅包含 "data"
# 3. Software/ML Engineer: 仅包含 "engineer"
# 4. Other: 不包含 "data" 或 "engineer"
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

# 应用分类规则 / Apply Classification Rules
df['Job Category'] = df['cleaned_job_title'].apply(categorize_job)

# 统计各类别的职位数量 / Count Job Categories
job_category_counts = df['Job Category'].value_counts()

# 解析薪资字段 / Parse Salary Data
def extract_salary(salary_str):
    if isinstance(salary_str, str):
        numbers = re.findall(r'\d+', salary_str)
        if len(numbers) >= 2:  # 取工资范围的均值 / Take the average of salary range
            return (int(numbers[0]) + int(numbers[1])) / 2
        elif len(numbers) == 1:
            return int(numbers[0])
    return np.nan

df['Salary (Processed)'] = df['Salary Estimate'].apply(extract_salary)

# 计算每个类别的平均薪资 / Compute Average Salary for Each Category
salary_by_category = df.groupby('Job Category')['Salary (Processed)'].mean()

# === 重新绘制职位类别分布柱状图 ===
plt.figure(figsize=(12, 6))  # 增大图表尺寸
sns.barplot(x=job_category_counts.index, y=job_category_counts.values, palette='Blues')
plt.xlabel('Job Category')
plt.ylabel('Number of Jobs')
plt.title('Job Distribution by Category')
plt.xticks(rotation=30, ha="right")  # 旋转 X 轴标签，避免重叠
plt.tight_layout()  # 自动调整布局，防止文字被裁剪
plt.show()

# === 重新绘制平均薪资柱状图 ===
plt.figure(figsize=(12, 6))  # 增大图表尺寸
sns.barplot(x=salary_by_category.index, y=salary_by_category.values, palette='viridis')
plt.xlabel('Job Category')
plt.ylabel('Average Salary ($1000s)')
plt.title('Average Salary by Job Category')
plt.xticks(rotation=30, ha="right")  # 旋转 X 轴标签，避免重叠
plt.tight_layout()  # 自动调整布局，防止文字被裁剪
plt.show()
