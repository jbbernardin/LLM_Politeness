import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the JSON data
with open('perplexity_results.json', 'r') as file:
    data = json.load(file)

# Prepare the data for plotting
records = []
for entry in data:
    for question_type in ['question', 'polite_question', 'impolite_question']:
        if question_type in entry and 'perplexity' in entry[question_type]:
            perplexity = entry[question_type]['perplexity']
            num_tokens = len(entry[question_type]['tokens'])
            perplexity_per_token = perplexity / num_tokens
            records.append({
                'Question Type': question_type.replace('_', ' ').title(),
                'Perplexity Per Token': perplexity_per_token,
                'num_tokens': num_tokens
            })

# Convert to DataFrame
df = pd.DataFrame(records)

# Set plot style
sns.set_style('whitegrid')

# Plot the distribution of perplexity per token by question type
plt.figure(figsize=(10,7))
for question_type in df['Question Type'].unique():
    sns.kdeplot(
        data=df[df['Question Type'] == question_type], 
        x='Perplexity Per Token', 
        label=question_type, 
        common_norm=False, 
        fill=True, 
        alpha=0.5
    )
plt.title('Distribution of Perplexity Per Token by Question Type')
plt.xlabel('Perplexity Per Token')
plt.ylabel('Density')
plt.legend(title='Question Type')
plt.show()

import matplotlib.ticker as ticker

# Plot the box and whiskers plot of perplexity per token by question type with a logarithmic scale
plt.figure(figsize=(10,7))
sns.boxplot(x='Question Type', y='Perplexity Per Token', data=df)
plt.yscale('log')
plt.title('Box and Whiskers Plot of Perplexity Per Token by Question Type')
plt.xlabel('Question Type')
plt.ylabel('Perplexity Per Token (log scale)')
plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter())
plt.show()

# Load the CSV data
evaluation_df = pd.read_csv('evaluation_results.csv')

# Prepare the perplexity data for merging
perplexity_records = []
for entry in data:
    for question_type in ['question', 'polite_question', 'impolite_question']:
        if question_type in entry and 'perplexity' in entry[question_type]:
            perplexity = entry[question_type]['perplexity']
            num_tokens = len(entry[question_type]['tokens'])
            perplexity_per_token = perplexity / num_tokens
            perplexity_records.append({
                'id': entry.get('id', ''),
                'answer_type': question_type + '_answer',
                'perplexity_per_token': perplexity_per_token,
                'question': entry[question_type].get('question', '')
            })

# Convert to DataFrame
perplexity_df = pd.DataFrame(perplexity_records)

# Merge the evaluation data with the perplexity data
merged_df = pd.merge(evaluation_df, perplexity_df, on=['question', 'answer_type'])

# Set plot style
sns.set_style('whitegrid')

# Plot correlation between perplexity per token and F1 score
plt.figure(figsize=(10, 7))
sns.scatterplot(data=merged_df, x='perplexity_per_token', y='F1', hue='answer_type')
plt.title('Correlation between Perplexity Per Token and F1 Score')
plt.xlabel('Perplexity Per Token')
plt.ylabel('F1 Score')
plt.xscale('log')
plt.legend(title='Question Type')
plt.show()

# Plot correlation between perplexity per token and contains score
plt.figure(figsize=(10, 7))
sns.scatterplot(data=merged_df, x='perplexity_per_token', y='contains', hue='answer_type')
plt.title('Correlation between Perplexity Per Token and Contains Score')
plt.xlabel('Perplexity Per Token')
plt.ylabel('Contains Score')
plt.xscale('log')
plt.legend(title='Question Type')
plt.show()

# Calculate the correlation coefficient between perplexity per token and contains score
correlation = merged_df[['perplexity_per_token', 'contains']].corr(method='pearson').iloc[0, 1]
print(f'Correlation coefficient between perplexity per token and contains score: {correlation}')

# Calculate correlation coefficient between perplexity per token and F1 score
correlation = merged_df[['perplexity_per_token', 'F1']].corr(method='pearson').iloc[0, 1]
print(f'Correlation coefficient between perplexity per token and F1 score: {correlation}')