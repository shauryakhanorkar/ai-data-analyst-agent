from src.data_loader import load_data
from src.metrics_engine import calculate_metric_answer

df = load_data()

questions = [
    "Show monthly sales",
    "What is the monthly sales trend?",
    "How did sales change month by month?"
]

for question in questions:
    print("\nQuestion:", question)

    try:
        result = calculate_metric_answer(question, df)
        print("Result:", result)
    except Exception as error:
        print("Error:", error)