"""Standard evaluation prompts covering diverse domains."""

KNOWLEDGE_PROMPTS = [
    "Explain the theory of general relativity in simple terms.",
    "What are the main differences between classical and quantum mechanics?",
    "Describe the process of photosynthesis step by step.",
    "What is the significance of the Turing test in artificial intelligence?",
    "Explain how CRISPR gene editing works.",
]

CODING_PROMPTS = [
    "Write a Python function to find the longest common subsequence of two strings.",
    "Implement a binary search algorithm in Python.",
    "Write a Python function that checks if a string is a valid palindrome.",
    "Explain the difference between a stack and a queue, with code examples.",
    "Write a Python function to merge two sorted lists.",
]

MATH_PROMPTS = [
    "What is the derivative of x^3 + 2x^2 - 5x + 1?",
    "Explain the Pythagorean theorem and give an example.",
    "What is the integral of sin(x) cos(x)?",
    "Solve the system of equations: 2x + y = 5, x - y = 1.",
    "Explain what a prime number is and list the first 10 primes.",
]

REASONING_PROMPTS = [
    "If all cats are animals, and all animals need food, do cats need food? Explain your reasoning.",
    "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
    "What comes next in the sequence: 2, 6, 12, 20, 30, ?",
    "If you have a 3-gallon jug and a 5-gallon jug, how do you measure exactly 4 gallons?",
]

SUMMARIZATION_PROMPTS = [
    "Summarize the key principles of democracy in three sentences.",
    "Summarize the plot of Romeo and Juliet in one paragraph.",
    "Explain the causes of World War I in a brief summary.",
    "Summarize the main ideas behind machine learning.",
    "Provide a brief summary of the water cycle.",
]

DIALOGUE_PROMPTS = [
    "User: What's the weather like today?\nAssistant:",
    "User: Can you help me write a resume?\nAssistant:",
    "User: I'm feeling stressed about my exam.\nAssistant:",
    "User: What's a good recipe for pasta?\nAssistant:",
    "User: How do I learn a new language effectively?\nAssistant:",
]

ALL_PROMPTS = {
    "knowledge": KNOWLEDGE_PROMPTS,
    "coding": CODING_PROMPTS,
    "math": MATH_PROMPTS,
    "reasoning": REASONING_PROMPTS,
    "summarization": SUMMARIZATION_PROMPTS,
    "dialogue": DIALOGUE_PROMPTS,
}

DEFAULT_PROMPTS = (
    KNOWLEDGE_PROMPTS[:1] +
    CODING_PROMPTS[:1] +
    MATH_PROMPTS[:1] +
    REASONING_PROMPTS[:1] +
    SUMMARIZATION_PROMPTS[:1] +
    DIALOGUE_PROMPTS[:1]
)


def get_prompts(categories=None, per_category=1):
    if categories is None:
        categories = list(ALL_PROMPTS.keys())
    prompts = []
    for cat in categories:
        if cat in ALL_PROMPTS:
            prompts.extend(ALL_PROMPTS[cat][:per_category])
    return prompts
