import json
import argparse
from tqdm import tqdm
import os
import torch

# Import model classes
from model_mistral import MistralModel
from model_falcon import FalconModel
from model_llama import LlamaModel

def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def check_correctness(predicted_answer, correct_answers):
    # Simple substring match
    for ans in correct_answers:
        if ans.lower() in predicted_answer.lower():
            return True
    return False

def main(args):
    data = load_data(args.data_path)

    # Initialize model
    if args.model_name == 'mistral':
        model = MistralModel(device=args.device)
    elif args.model_name == 'falcon':
        model = FalconModel(device=args.device)
    elif args.model_name == 'llama':
        model = LlamaModel(device=args.device)
    else:
        raise ValueError("Invalid model name!")

    question_types = [
        "question",
        "neutral_question_1", "neutral_question_2", "neutral_question_3",
        "polite_question_1", "polite_question_2", "polite_question_3",
        "impolite_question_1", "impolite_question_2", "impolite_question_3"
    ]

    # Initialize counters for each question type
    results = {q_type: {"correct": 0, "total": 0} for q_type in question_types}

    for item in tqdm(data, desc="Evaluating"):
        context = item['context']
        correct_answers = item['answers']['text']
        for q_type in question_types:
            if q_type in item:
                question = item[q_type]
                predicted = model.generate_answer(context, question)
                # Check correctness
                is_correct = check_correctness(predicted, correct_answers)
                if is_correct:
                    results[q_type]["correct"] += 1
                results[q_type]["total"] += 1

    # Compute accuracy for each question type
    per_type_accuracy = {}
    total_correct = 0
    total_count = 0
    for q_type, counts in results.items():
        if counts["total"] > 0:
            acc = (counts["correct"] / counts["total"]) * 100
        else:
            acc = 0.0
        per_type_accuracy[q_type] = acc
        total_correct += counts["correct"]
        total_count += counts["total"]

    overall_accuracy = (total_correct / total_count) * 100 if total_count > 0 else 0

    print(f"Model: {args.model_name}, Overall Accuracy: {overall_accuracy:.2f}%\n")
    print("Per Question Type Accuracy:")
    for q_type, acc in per_type_accuracy.items():
        print(f"{q_type}: {acc:.2f}%")

    
    output_data = {
        "model": args.model_name,
        "overall_accuracy": overall_accuracy,
        "per_question_type_accuracy": per_type_accuracy
    }

    with open("evaluation_results.json", "w") as outfile:
        json.dump(output_data, outfile, indent=4)

    print("\nResults saved to evaluation_results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="filtered_politeness_modified.json")
    parser.add_argument("--model_name", type=str, required=True, choices=['mistral','falcon','llama'])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)
