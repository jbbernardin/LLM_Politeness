import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

# Load the CSV data
df = pd.read_csv('evaluation_results.csv')

# Pivot the data to have one row per question with columns for each answer type
pivot_df = df.pivot(index='id', columns='answer_type', values='contains')

# Define a function to categorize the questions
def categorize_question(row):
    if row['question_answer'] and row['polite_question_answer'] and row['impolite_question_answer']:
        return 'All Correct'
    elif not row['question_answer'] and not row['polite_question_answer'] and not row['impolite_question_answer']:
        return 'All Incorrect'
    elif row['question_answer'] and not row['polite_question_answer'] and not row['impolite_question_answer']:
        return 'Original'
    elif not row['question_answer'] and row['polite_question_answer'] and not row['impolite_question_answer']:
        return 'Polite'
    elif not row['question_answer'] and not row['polite_question_answer'] and row['impolite_question_answer']:
        return 'Impolite'
    elif row['question_answer'] and row['polite_question_answer'] and not row['impolite_question_answer']:
        return 'Original / Polite'
    elif row['question_answer'] and not row['polite_question_answer'] and row['impolite_question_answer']:
        return 'Original / Impolite'
    elif not row['question_answer'] and row['polite_question_answer'] and row['impolite_question_answer']:
        return 'Polite / Impolite'
    else:
        return 'Error'

# Apply the function to categorize each question
pivot_df['Category'] = pivot_df.apply(categorize_question, axis=1)

# Count the number of questions in each category
category_counts = pivot_df['Category'].value_counts()

# Calculate the percentage for each category
category_percentages = (category_counts / category_counts.sum())

# Wrap the x-axis labels
wrapped_labels = [textwrap.fill(label, 10) for label in category_percentages.index]

# Plot the distribution of question categories as percentages
plt.figure(figsize=(10, 7))
sns.barplot(x=wrapped_labels, y=category_percentages.values, palette='viridis')
plt.title('Distribution of Correctness by Politeness')
plt.xlabel('Correct Type')
plt.ylabel('Percentage of Questions')
plt.ylim(0, 1)  # Set y-axis limit to 100%
plt.show()

# Print the percentages for each category
print(category_percentages)