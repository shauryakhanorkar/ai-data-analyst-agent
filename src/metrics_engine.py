
import pandas as pd


def normalize_question(question):
    """Convert common business phrases into consistent keywords."""

    text = question.lower().strip()

    replacements = {
        "revenues": "sales",
        "revenue": "sales",
        "turnover": "sales",
        "income": "profit",
        "earnings": "profit",
        "mean value": "average",
        "on average": "average",
        "most profitable": "highest profit",
        "least profitable": "lowest profit",
        "how much did we make": "total sales",
        "how much have we made": "total sales",
        "total amount of sales": "total sales",
        "sum of sales": "total sales",
        "sum of profit": "total profit",
        "number of orders": "how many orders",
        "count of orders": "how many orders",
        "number of records": "how many records",
        "number of rows": "how many rows",
    }

    for phrase, replacement in replacements.items():
        text = text.replace(phrase, replacement)

    return text


def find_column(df, possible_names):
    """Find a column using case-insensitive name matching."""

    column_lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for name in possible_names:
        actual_column = column_lookup.get(name.lower())

        if actual_column is not None:
            return actual_column

    return None


def filter_data_from_question(question, df):
    """Filter data by a category or region mentioned in the question."""

    question_lower = question.lower()
    filtered_df = df.copy()
    applied_filters = []

    category_col = find_column(
        df, ["category", "product category"]
    )
    region_col = find_column(df, ["region"])

    # Filter by category.
    if category_col:
        for category in df[category_col].dropna().unique():
            if str(category).lower() in question_lower:
                filtered_df = filtered_df[
                    filtered_df[category_col] == category
                ]
                applied_filters.append(f"Category: {category}")
                break

    # Filter by region.
    if region_col:
        for region in df[region_col].dropna().unique():
            if str(region).lower() in question_lower:
                filtered_df = filtered_df[
                    filtered_df[region_col] == region
                ]
                applied_filters.append(f"Region: {region}")
                break

    return filtered_df, applied_filters


def calculate_metric_answer(question, df):
    """
    Calculate supported business metrics directly from a DataFrame.

    Returns a dictionary for supported questions, or None if the
    question is not recognized.
    """

    if df is None or df.empty:
        raise ValueError("The dataset is empty.")

    original_df = df.copy()

    df, applied_filters = filter_data_from_question(question, df)

    if df.empty:
        raise ValueError("No data matches the filters in your question.")

    question_lower = normalize_question(question)

    sales_col = find_column(df, ["sales", "total sales"])
    profit_col = find_column(df, ["profit"])
    category_col = find_column(df, ["category", "product category"])
    region_col = find_column(df, ["region"])
    date_col = find_column(
        df, ["order date", "date", "order_date"]
    )
    order_col = find_column(
        df, ["order id", "order_id", "order number", "order no"]
    )

    # Prepare numeric columns.
    sales = (
        pd.to_numeric(df[sales_col], errors="coerce").dropna()
        if sales_col else None
    )
    profit = (
        pd.to_numeric(df[profit_col], errors="coerce").dropna()
        if profit_col else None
    )

    # ---------------------------------------------------------
    # 1. Count rows.
    # ---------------------------------------------------------
    if any(word in question_lower for word in [
        "how many rows", "number of rows", "row count",
        "how many records", "number of records"
    ]):
        return {
            "metric": "row_count",
            "result": int(len(df)),
            "meaning": "Number of rows in the selected data"
        }

    # ---------------------------------------------------------
    # 2. Count orders.
    # ---------------------------------------------------------
    if (
        "order" in question_lower
        and any(word in question_lower for word in [
            "how many", "count", "number of", "total"
        ])
    ):
        if order_col:
            result = int(df[order_col].nunique(dropna=True))
            method = "Number of distinct order IDs"
        else:
            result = int(len(df))
            method = (
                "Rows counted because no order ID column was found. "
                "This may not equal the number of unique orders."
            )

        return {
            "metric": "order_count",
            "result": result,
            "meaning": method
        }

    # ---------------------------------------------------------
    # 3. Identify the requested numeric metric.
    # ---------------------------------------------------------
    if any(word in question_lower for word in ["sales", "revenue"]):
        metric_name = "sales"
        column = sales_col
        values = sales
    elif "profit" in question_lower:
        metric_name = "profit"
        column = profit_col
        values = profit
    else:
        metric_name = None
        column = None
        values = None

    # ---------------------------------------------------------
    # 4. Monthly sales analysis.
    # IMPORTANT: This must run BEFORE maximum individual sales.
    # ---------------------------------------------------------
    monthly_question_phrases = [
        "monthly",
        "month by month",
        "monthly trend",
        "sales trend",
        "over time",
        "which month",
        "what month",
        "month had the highest",
        "month had highest",
        "month has the highest",
        "highest sales month",
        "highest selling month",
        "best month for sales",
        "sales by month",
        "sales per month",
    ]

    is_monthly_question = any(
        phrase in question_lower
        for phrase in monthly_question_phrases
    )

    if (
        is_monthly_question
        and metric_name == "sales"
        and sales_col
        and date_col
    ):
        trend_df = df[[date_col, sales_col]].copy()

        trend_df[date_col] = pd.to_datetime(
            trend_df[date_col], errors="coerce"
        )
        trend_df[sales_col] = pd.to_numeric(
            trend_df[sales_col], errors="coerce"
        )

        trend_df = trend_df.dropna(
            subset=[date_col, sales_col]
        )

        trend_df["Month"] = (
            trend_df[date_col].dt.to_period("M").astype(str)
        )

        monthly_sales = (
            trend_df.groupby("Month")[sales_col]
            .sum()
            .sort_index()
        )

        if not monthly_sales.empty:
            monthly_results = {
                month: round(float(total), 2)
                for month, total in monthly_sales.items()
            }

            highest_month = monthly_sales.idxmax()

            changes = {}

            for i in range(1, len(monthly_sales)):
                previous = monthly_sales.iloc[i - 1]
                current = monthly_sales.iloc[i]
                previous_month = monthly_sales.index[i - 1]
                current_month = monthly_sales.index[i]

                changes[current_month] = {
                    "difference": round(
                        float(current - previous), 2
                    ),
                    "percentage_change": (
                        round(
                            float(
                                (current - previous)
                                / abs(previous) * 100
                            ),
                            2
                        )
                        if previous != 0 else None
                    )
                }

            return {
                "metric": "monthly_sales_trend",
                "monthly_sales": monthly_results,
                "highest_sales_month": highest_month,
                "highest_month_sales": round(
                    float(monthly_sales.max()), 2
                ),
                "month_over_month_changes": changes,
                "meaning": (
                    "Sales totals grouped by calendar month. "
                    "The highest-sales month is based on the sum "
                    "of all sales within each month. Changes compare "
                    "each month with the previous month."
                ),
                "filters_applied": applied_filters
            }

    # ---------------------------------------------------------
    # 5. Percentage of total sales or profit.
    # ---------------------------------------------------------
    is_percentage_question = any(
        word in question_lower
        for word in ["percentage", "percent", "%", "share of"]
    )

    is_comparison_question = any(
        word in question_lower
        for word in [
            "compare", "comparison", "versus", " vs ",
            "difference between", "between"
        ]
    )

    if (
        metric_name
        and column
        and is_percentage_question
        and not is_comparison_question
        and applied_filters
    ):
        filtered_total = pd.to_numeric(
            df[column], errors="coerce"
        ).sum()

        overall_total = pd.to_numeric(
            original_df[column], errors="coerce"
        ).sum()

        if overall_total != 0:
            return {
                "metric": f"{metric_name}_percentage_of_total",
                "group": applied_filters,
                "result": round(
                    float(filtered_total / overall_total * 100), 2
                ),
                "meaning": (
                    f"Percentage of overall {metric_name} "
                    "represented by the selected data"
                )
            }

    # ---------------------------------------------------------
    # 6. Compare two categories or regions.
    # ---------------------------------------------------------
    if is_comparison_question and metric_name and column:
        group_columns = [
            (category_col, "category"),
            (region_col, "region")
        ]

        for group_col, group_name in group_columns:
            if group_col is None:
                continue

            mentioned_groups = [
                value
                for value in original_df[group_col].dropna().unique()
                if str(value).lower() in question_lower
            ]

            if len(mentioned_groups) >= 2:
                comparison_df = original_df[
                    original_df[group_col].isin(mentioned_groups)
                ].copy()

                comparison_df[column] = pd.to_numeric(
                    comparison_df[column], errors="coerce"
                )

                grouped = comparison_df.groupby(
                    group_col
                )[column].sum()

                results = {
                    str(group): round(float(total), 2)
                    for group, total in grouped.items()
                }

                if len(results) >= 2:
                    first_group, second_group = list(results.keys())[:2]
                    first_value = results[first_group]
                    second_value = results[second_group]

                    difference = round(
                        abs(first_value - second_value), 2
                    )

                    percentage_change = (
                        round(
                            (second_value - first_value)
                            / abs(first_value) * 100,
                            2
                        )
                        if first_value != 0 else None
                    )

                    return {
                        "metric": f"{metric_name}_comparison",
                        "groups": results,
                        "difference": difference,
                        "percentage_change_from_first": (
                            percentage_change
                        ),
                        "meaning": (
                            f"Total {metric_name} comparison between "
                            f"{group_name}s. Percentage change uses "
                            f"{first_group} as the reference group."
                        )
                    }

    # ---------------------------------------------------------
    # 7. Average and median.
    # ---------------------------------------------------------
    if values is not None and column is not None and len(values) > 0:

        if any(word in question_lower for word in [
            "average", "mean", "on average"
        ]):
            return {
                "metric": f"average_{metric_name}",
                "column": column,
                "result": round(float(values.mean()), 2),
                "meaning": f"Average {metric_name} per non-missing value"
            }

        if "median" in question_lower:
            return {
                "metric": f"median_{metric_name}",
                "column": column,
                "result": round(float(values.median()), 2),
                "meaning": f"Median {metric_name} per non-missing value"
            }

        # -----------------------------------------------------
        # 8. Highest and lowest individual values.
        # These checks now happen AFTER monthly analysis.
        # -----------------------------------------------------
        if (
            any(word in question_lower for word in [
                "highest", "maximum", "max", "largest"
            ])
            and not any(word in question_lower for word in [
                "category", "region"
            ])
        ):
            return {
                "metric": f"maximum_{metric_name}",
                "column": column,
                "result": round(float(values.max()), 2),
                "meaning": f"Highest individual {metric_name} value"
            }

        if (
            any(word in question_lower for word in [
                "lowest", "minimum", "min", "smallest"
            ])
            and not any(word in question_lower for word in [
                "category", "region"
            ])
        ):
            return {
                "metric": f"minimum_{metric_name}",
                "column": column,
                "result": round(float(values.min()), 2),
                "meaning": f"Lowest individual {metric_name} value"
            }

        # -----------------------------------------------------
        # 9. Total sales or profit.
        # -----------------------------------------------------
        if (
            any(word in question_lower for word in [
                "total", "sum", "overall"
            ])
            or not any(word in question_lower for word in [
                "average", "median", "highest", "maximum", "max",
                "largest", "lowest", "minimum", "min", "smallest",
                "category", "region", " by ", "percentage",
                "percent", "share of", "compare", "comparison",
                "versus", " vs ", "between"
            ])
        ):
            return {
                "metric": f"total_{metric_name}",
                "column": column,
                "result": round(float(values.sum()), 2),
                "meaning": f"Sum of all non-missing {metric_name} values"
            }

    # ---------------------------------------------------------
    # 10. Highest or lowest sales/profit by category or region.
    # ---------------------------------------------------------
    if metric_name and values is not None and column is not None:
        group_col = None

        if "category" in question_lower and category_col:
            group_col = category_col
        elif "region" in question_lower and region_col:
            group_col = region_col

        if group_col is not None:
            temp = df[[group_col, column]].copy()
            temp[column] = pd.to_numeric(
                temp[column], errors="coerce"
            )

            grouped = temp.groupby(
                group_col, dropna=True
            )[column].sum().dropna()

            if not grouped.empty:
                descending = not any(
                    word in question_lower
                    for word in [
                        "lowest", "least", "minimum", "smallest"
                    ]
                )

                selected_group = (
                    grouped.idxmax() if descending else grouped.idxmin()
                )

                return {
                    "metric": f"{metric_name}_by_{group_col}",
                    "group": str(selected_group),
                    "result": round(
                        float(grouped.loc[selected_group]), 2
                    ),
                    "meaning": (
                        f"{metric_name.title()} summed by {group_col}; "
                        f"{'highest' if descending else 'lowest'} group"
                    )
                }

    # ---------------------------------------------------------
    # 11. Count missing values.
    # ---------------------------------------------------------
    if any(word in question_lower for word in [
        "missing values", "null values", "empty values",
        "missing data", "null data"
    ]):
        return {
            "metric": "missing_values",
            "result": int(original_df.isna().sum().sum()),
            "meaning": "Total missing cells across the dataset"
        }

    # ---------------------------------------------------------
    # 12. Count duplicate rows.
    # ---------------------------------------------------------
    if any(word in question_lower for word in [
        "duplicate rows", "duplicate records",
        "how many duplicates", "count duplicates"
    ]):
        return {
            "metric": "duplicate_rows",
            "result": int(original_df.duplicated().sum()),
            "meaning": "Number of rows duplicated across all columns"
        }

    # ---------------------------------------------------------
    # 13. Count distinct categories.
    # ---------------------------------------------------------
    if (
        category_col
        and "categor" in question_lower
        and any(word in question_lower for word in [
            "how many", "count", "number of", "unique", "different"
        ])
    ):
        return {
            "metric": "unique_categories",
            "result": int(df[category_col].nunique(dropna=True)),
            "meaning": "Number of distinct non-missing categories"
        }

    return None
