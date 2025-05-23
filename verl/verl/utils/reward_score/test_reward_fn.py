import re

def extract_answer_number(text: str) -> int | None:
    """
    Extract the first integer following the first occurrence of
    'Answer:', 'ANSWER:', etc. Returns None if no integer is found.
    """
    m = re.search(r'answer\s*:\s*(.+)', text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        answer_text = m.group(1).strip()
        num_match = re.search(r'-?\d+', answer_text)
        if num_match:
            return num_match.group()
    return None

def acc_reward(predict_str: str, ground_truth: str) -> float:
    extracted_num = extract_answer_number(predict_str)
    return 1.0 if extracted_num == ground_truth else 0.0

def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    return acc_reward(solution_str, ground_truth)