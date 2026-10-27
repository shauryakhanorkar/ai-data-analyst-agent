import pandas as pd
try:
    from .data_loader import load_data
except ImportError:
    from data_loader import load_data


def get_dataset_overview(df):
    """Return basic information about the dataset."""

    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "data_types": df.dtypes.astype(str).to_dict(),
    }


def get_data_quality_report(df):
    """Check missing values and duplicate rows."""

    missing_count = df.isnull().sum()

    report = pd.DataFrame({
        "Missing Values": missing_count,
        "Missing Percentage": (
            missing_count / len(df) * 100
        ).round(2),
        "Data Type": df.dtypes.astype(str),
    })

    return {
        "report": report,
        "duplicate_rows": int(df.duplicated().sum()),
        "empty_rows": int(df.isnull().all(axis=1).sum()),
    }


def get_numeric_summary(df):
    """Calculate statistics for numeric columns."""

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        return pd.DataFrame()

    return numeric_df.describe().round(2)


def get_business_insights(df):
    """Calculate sales and profit insights."""

    insights = {}

    # Match column names without case or whitespace issues.
    column_lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    sales_col = column_lookup.get("sales")
    profit_col = column_lookup.get("profit")
    category_col = column_lookup.get("category")
    region_col = column_lookup.get("region")

    # Total and average sales
    if sales_col:
        sales = pd.to_numeric(df[sales_col], errors="coerce")

        insights["total_sales"] = round(sales.sum(), 2)

        if sales.notna().any():
            insights["average_sale"] = round(sales.mean(), 2)

    # Total and average profit
    if profit_col:
        profit = pd.to_numeric(df[profit_col], errors="coerce")

        insights["total_profit"] = round(profit.sum(), 2)

        if profit.notna().any():
            insights["average_profit"] = round(profit.mean(), 2)

    # Sales by category
    if category_col and sales_col:
        category_data = df[[category_col, sales_col]].copy()
        category_data[sales_col] = pd.to_numeric(
            category_data[sales_col], errors="coerce"
        )

        category_sales = category_data.groupby(
            category_col, dropna=False
        )[sales_col].sum()

        insights["sales_by_category"] = (
            category_sales.sort_values(ascending=False).round(2)
        )

        if category_sales.notna().any():
            insights["top_category_by_sales"] = (
                category_sales.idxmax()
            )

    # Profit by category
    if category_col and profit_col:
        category_profit = df.groupby(
            category_col, dropna=False
        )[profit_col].sum()

        insights["profit_by_category"] = (
            category_profit.sort_values(ascending=False).round(2)
        )

    # Sales by region
    if region_col and sales_col:
        region_data = df[[region_col, sales_col]].copy()
        region_data[sales_col] = pd.to_numeric(
            region_data[sales_col], errors="coerce"
        )

        region_sales = region_data.groupby(
            region_col, dropna=False
        )[sales_col].sum()

        insights["sales_by_region"] = (
            region_sales.sort_values(ascending=False).round(2)
        )

        if region_sales.notna().any():
            insights["top_region_by_sales"] = region_sales.idxmax()

    return insights


if __name__ == "__main__":
    df = load_data()

    print("\n========== DATASET OVERVIEW ==========")

    overview = get_dataset_overview(df)

    print(f"Total rows: {overview['total_rows']}")
    print(f"Total columns: {overview['total_columns']}")
    print(f"Columns: {overview['column_names']}")

    print("\nData types:")
    for column, dtype in overview["data_types"].items():
        print(f"{column}: {dtype}")

    print("\n========== DATA QUALITY ==========")

    quality = get_data_quality_report(df)

    print(quality["report"].to_string())
    print(f"\nDuplicate rows: {quality['duplicate_rows']}")
    print(f"Completely empty rows: {quality['empty_rows']}")

    print("\n========== NUMERIC SUMMARY ==========")

    print(get_numeric_summary(df).to_string())

    print("\n========== BUSINESS INSIGHTS ==========")

    insights = get_business_insights(df)

    for name, value in insights.items():
        print(f"\n{name.replace('_', ' ').title()}:")

        if isinstance(value, pd.Series):
            print(value.to_string())
        else:
            print(value)