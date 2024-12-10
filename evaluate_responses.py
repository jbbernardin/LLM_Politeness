import json
import csv
import re
from collections import Counter

# Function to compute Exact Match
def exact_match(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

# Function to compute F1 Score
def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

# Function to check if ground truth is contained in prediction
def is_contained(prediction, ground_truth):
    return normalize_answer(ground_truth) in normalize_answer(prediction)

# Function to normalize text
def normalize_answer(s):
    """Lower text and remove punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def remove_punctuation(text):
        return re.sub(r'[^\w\s]', '', text)

    def lowercase(text):
        return text.lower()

    def white_space_fix(text):
        return ' '.join(text.split())

    return white_space_fix(remove_articles(remove_punctuation(lowercase(s))))

# Load the JSON data
with open('4o-mini_answers.json', 'r') as file:
    data = json.load(file)

answer_types = [
    "question_answer", 
    "neutral_answer_1", 
    "neutral_answer_2", 
    "neutral_answer_3", 
    "polite_answer_1",
    "polite_answer_2",
    "polite_answer_3",
    "impolite_answer_1",
    "impolite_answer_2",
    "impolite_answer_3"
]


# Prepare CSV file
with open('4o-mini_evaluation_results.csv', 'w', newline='') as csvfile:
    fieldnames = ['id', 'question', 'answer_type', 'EM', 'F1', 'contains']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Iterate over each entry in the data
    for entry in data:
        q_id = entry.get('id', '')
        question = entry.get('question', '')
        ground_truth = entry.get('ground_truth_answer', '')
        print(ground_truth)

        for ans_type in answer_types:
            prediction = entry.get(ans_type, '')
            if prediction:
                em = exact_match(prediction, ground_truth)
                f1 = f1_score(prediction, ground_truth)
                contains = is_contained(prediction, ground_truth)
                question_type = "question" if "question_answer" == ans_type else ans_type.replace("answer", "question")
                writer.writerow({
                    'id': q_id,
                    'answer_type': ans_type,
                    'EM': em,
                    'F1': f1,
                    'contains': contains,
                    'question': entry[question_type]
                })
            else:
                print(f"Missing prediction for {q_id} - {ans_type}")

print("Evaluation completed. Results saved to evaluation_results.csv")