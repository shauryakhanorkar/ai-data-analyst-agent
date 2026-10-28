try:
    from .data_loader import load_data
    from .data_analyzer import get_business_insights
    from .llm_client import ask_gemini
    from .metrics_engine import calculate_metric_answer

except ImportError:
    from data_loader import load_data
    from data_analyzer import get_business_insights
    from llm_client import ask_gemini
    from metrics_engine import calculate_metric_answer


def answer_question(question, df=None):
    """
    Answer a business question using Python-calculated metrics
    and Gemini for natural-language explanations.
    """

    if not question or not question.strip():
        raise ValueError("Please enter a question.")

    if df is None:
        df = load_data()

    if df is None or df.empty:
        raise ValueError("The dataset is empty or could not be loaded.")

    # Step 1: Calculate general business insights.
    insights = get_business_insights(df)

    if not insights:
        raise ValueError("No business insights could be calculated.")

    # Step 2: Calculate the direct answer using Python.
    metric_result = calculate_metric_answer(question, df)

    # Step 3: Prepare dataset information.
    columns = list(df.columns)
    row_count = len(df)

    # Step 4: Give Gemini the calculated result and clear instructions.
    prompt = f"""
You are an AI Data Analyst helping a business user understand
their sales dataset.

DATASET INFORMATION
Number of rows: {row_count}
Column names: {columns}

GENERAL BUSINESS INSIGHTS CALCULATED BY PYTHON
{insights}

DIRECT METRIC RESULT CALCULATED BY PYTHON
{metric_result if metric_result is not None else "No matching direct metric was calculated."}

USER QUESTION
{question}

IMPORTANT INSTRUCTIONS

1. First, examine the DIRECT METRIC RESULT calculated by Python.

2. If the direct metric result contains the answer to the user's
   question, you MUST use that result.

3. The direct metric result has priority over the general business
   insights. The general business insights may not contain every
   calculation performed by Python.

4. For questions about monthly sales, monthly trends, or the month
   with the highest sales:
   - Check whether the direct metric result contains monthly sales
     information.
   - If it contains a highest-sales month, report that month and
     its calculated sales value.
   - If it contains monthly sales values but no explicit
     highest-sales month, identify the month with the largest
     sales value from those results.
   - Do not claim that monthly information is unavailable when
     the direct metric result contains it.

5. For questions about totals, averages, comparisons, percentages,
   categories, regions, and profit, use the corresponding values
   from the direct metric result whenever available.

6. Do not invent numbers, months, categories, or regions.

7. Do not recalculate or contradict a valid Python-calculated result.

8. If the direct metric result is None or does not contain enough
   information to answer the question, explain the limitation
   honestly. You may then use the general business insights if
   they contain the answer.

9. Answer in simple, professional English.

10. Keep the answer concise and directly address the question.

11. Format monetary values clearly, using two decimal places when
    appropriate.
"""

    # Step 5: Ask Gemini to explain the result.
    return ask_gemini(prompt)


def generate_business_summary(df=None):
    """
    Generate an AI-written business summary from Python-calculated
    sales and profit insights.
    """

    if df is None:
        df = load_data()

    if df is None or df.empty:
        raise ValueError("The dataset is empty or could not be loaded.")

    insights = get_business_insights(df)

    if not insights:
        raise ValueError("No business insights could be calculated.")

    prompt = f"""
You are a professional business data analyst.

Analyze the following calculated business insights and write a clear,
well-formatted business performance summary.

CALCULATED BUSINESS INSIGHTS:
{insights}

IMPORTANT FORMATTING RULES:
1. Write in natural, professional English.
2. Include normal spaces between every word.
3. Put each paragraph on a separate line.
4. Use Markdown headings and bullet points.
5. Do not join words together.
6. Do not remove spaces around numbers, punctuation, or currency values.
7. Format numbers clearly, for example: $2,297,200.86.
8. Do not put the entire response in one line.
9. Do not use HTML tags.
10. Do not invent or change any calculated figures.

Use exactly these sections:

## Overall Performance Summary
Write one short paragraph about total sales, average sale value,
total profit, and average profit if those figures are available.

## Sales and Profit Findings
- Category performance
- Regional performance

## Practical Recommendations
Give up to three actionable recommendations supported by the data.

## Important Limitations
Mention what cannot be concluded from the available dataset.

Return only the formatted Markdown summary.
"""

    return ask_gemini(prompt)

if __name__ == "__main__":

    print("\nAI Data Analyst")
    print("------------------------------")

    question = input("Ask a question about the sales data: ")

    if not question.strip():
        print("Please enter a question.")
    else:
        try:
            answer = answer_question(question)

            print("\nAI Analyst's Answer:")
            print("------------------------------")
            print(answer)

        except Exception as error:
            print(f"\nUnable to answer your question: {error}")