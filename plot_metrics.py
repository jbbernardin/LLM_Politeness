import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('evaluation_results.csv')

# Map 'answer_type' to more readable labels
answer_type_mapping = {
    'question_answer': 'Original Question',
    'polite_question_answer': 'Polite Question',
    'impolite_question_answer': 'Impolite Question'
}

df['Question Type'] = df['answer_type'].map(answer_type_mapping)

# Set plot style
sns.set_style('whitegrid')

# Plot overlapping distributions of F1 scores one at a time
plt.figure(figsize=(10,7))

for question_type, label in answer_type_mapping.items():
    subset = df[df['answer_type'] == question_type]
    sns.kdeplot(
        data=subset, 
        x='F1', 
        label=label, 
        common_norm=False, 
        fill=True, 
        alpha=0.3
    )

plt.title('Distribution of F1 Scores by Question Type')
plt.xlabel('F1 Score')
plt.ylabel('Density')
plt.legend(title='Question Type')
plt.show()

# Convert 'contains' column to numeric if necessary
df['contains'] = df['contains'].astype(int)

# Group data by 'answer_type' and calculate mean scores
grouped = df.groupby('answer_type', as_index=False).agg({
    'contains': 'mean'
})

# Set plot style
sns.set_style('whitegrid')

# Plot Contains scores
plt.figure(figsize=(8,6))
sns.barplot(x='answer_type', y='contains', data=grouped)
plt.title('Average Contains Score by Answer Type')
plt.xlabel('Answer Type')
plt.ylabel('Average Contains Score')
plt.show()