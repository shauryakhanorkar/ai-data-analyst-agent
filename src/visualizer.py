import matplotlib.pyplot as plt
import pandas as pd

from data_loader import load_data


def get_column(df, column_name):
    """Find a column without depending on capitalization or spaces."""

    for column in df.columns:
        if str(column).strip().lower() == column_name.lower():
            return column

    return None


def plot_sales_by_category(df):
    """Create a bar chart of total sales by category."""

    category_col = get_column(df, "Category")
    sales_col = get_column(df, "Sales")

    if category_col is None or sales_col is None:
        print("Cannot create category chart: required columns are missing.")
        return

    data = df[[category_col, sales_col]].copy()
    data[sales_col] = pd.to_numeric(data[sales_col], errors="coerce")
    data = data.dropna(subset=[category_col, sales_col])

    summary = data.groupby(category_col)[sales_col].sum().sort_values(
        ascending=False
    )

    if summary.empty:
        print("No valid data for sales by category.")
        return

    plt.figure(figsize=(9, 5))
    summary.plot(kind="bar")

    plt.title("Total Sales by Category")
    plt.xlabel("Category")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()


def plot_sales_by_region(df):
    """Create a bar chart of total sales by region."""

    region_col = get_column(df, "Region")
    sales_col = get_column(df, "Sales")

    if region_col is None or sales_col is None:
        print("Cannot create region chart: required columns are missing.")
        return

    data = df[[region_col, sales_col]].copy()
    data[sales_col] = pd.to_numeric(data[sales_col], errors="coerce")
    data = data.dropna(subset=[region_col, sales_col])

    summary = data.groupby(region_col)[sales_col].sum().sort_values(
        ascending=False
    )

    if summary.empty:
        print("No valid data for sales by region.")
        return

    plt.figure(figsize=(9, 5))
    summary.plot(kind="bar")

    plt.title("Total Sales by Region")
    plt.xlabel("Region")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()


def plot_profit_by_category(df):
    """Create a bar chart of total profit by category."""

    category_col = get_column(df, "Category")
    profit_col = get_column(df, "Profit")

    if category_col is None or profit_col is None:
        print("Cannot create profit chart: required columns are missing.")
        return

    data = df[[category_col, profit_col]].copy()
    data[profit_col] = pd.to_numeric(data[profit_col], errors="coerce")
    data = data.dropna(subset=[category_col, profit_col])

    summary = data.groupby(category_col)[profit_col].sum().sort_values(
        ascending=False
    )

    if summary.empty:
        print("No valid data for profit by category.")
        return

    plt.figure(figsize=(9, 5))
    summary.plot(kind="bar")

    plt.title("Total Profit by Category")
    plt.xlabel("Category")
    plt.ylabel("Total Profit")
    plt.axhline(y=0, linewidth=1)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()


def plot_monthly_sales(df):
    """Create a monthly sales line chart when an order-date column exists."""

    date_col = get_column(df, "Order Date")
    sales_col = get_column(df, "Sales")

    if date_col is None or sales_col is None:
        print("Cannot create monthly chart: Order Date or Sales is missing.")
        return

    data = df[[date_col, sales_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data[sales_col] = pd.to_numeric(data[sales_col], errors="coerce")
    data = data.dropna(subset=[date_col, sales_col])

    if data.empty:
        print("No valid dates or sales values for the monthly chart.")
        return

    data["Month"] = data[date_col].dt.to_period("M")

    monthly_sales = data.groupby("Month")[sales_col].sum().sort_index()
    monthly_sales.index = monthly_sales.index.astype(str)

    plt.figure(figsize=(11, 5))
    monthly_sales.plot(kind="line", marker="o")

    plt.title("Monthly Sales Trend")
    plt.xlabel("Month")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = load_data()

    print("Generating sales and profit charts...")

    plot_sales_by_category(df)
    plot_sales_by_region(df)
    plot_profit_by_category(df)
    plot_monthly_sales(df)

    print("Chart generation completed.")