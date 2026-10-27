
import pytest

from src.data_loader import load_data
from src.metrics_engine import calculate_metric_answer


@pytest.fixture(scope="module")
def df():
    return load_data()


def test_dataset_loads_successfully(df):
    assert df is not None
    assert not df.empty
    assert "Order Date" in df.columns
    assert "Sales" in df.columns


@pytest.mark.parametrize(
    "question",
    [
        "Show monthly sales",
        "What is the monthly sales trend?",
        "How did sales change month by month?",
    ],
)
def test_monthly_sales_questions(question, df):
    result = calculate_metric_answer(question, df)
    assert result is not None


def test_monthly_sales_calculation_matches_dataset(df):
    dates = __import__("pandas").to_datetime(
        df["Order Date"], errors="coerce"
    )

    monthly_sales = (
        df.assign(Month=dates.dt.to_period("M"))
        .dropna(subset=["Month"])
        .groupby("Month")["Sales"]
        .sum()
    )

    assert not monthly_sales.empty
    assert monthly_sales.sum() == pytest.approx(
        df.loc[dates.notna(), "Sales"].sum()
    )


def test_sales_values_are_numeric_and_finite(df):
    import numpy as np

    sales = df["Sales"].dropna()

    assert len(sales) > 0
    assert np.isfinite(sales).all()
