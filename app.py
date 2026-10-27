import streamlit as st
import pandas as pd
import plotly.express as px

from src.ai_analyst import answer_question

from src.data_analyzer import (
    get_dataset_overview,
    get_data_quality_report,
    get_numeric_summary,
    get_business_insights,
)

st.set_page_config(
    page_title="AI Data Analyst Agent",
    page_icon="📊",
    layout="wide",
)

st.title("📊 AI Data Analyst Agent")
st.write(
    "Upload a dataset to explore its data quality, "
    "business metrics, and visualizations."
)

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as error:
        st.error(f"Could not read the file: {error}")
        st.stop()

    if df.empty:
        st.error("The uploaded file contains no data.")
        st.stop()

    st.success("Dataset loaded successfully!")

    overview = get_dataset_overview(df)
    quality = get_data_quality_report(df)
    insights = get_business_insights(df)

    # KPI cards
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Rows", f"{overview['total_rows']:,}")
    col2.metric("Total Columns", overview["total_columns"])
    col3.metric(
        "Duplicate Rows",
        quality["duplicate_rows"],
    )

    # Data preview
    st.header("1. Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    # Data quality
    st.header("2. Data Quality Report")
    st.dataframe(quality["report"], use_container_width=True)

    st.write(
        f"**Completely empty rows:** {quality['empty_rows']}"
    )

    # Numeric statistics
    st.header("3. Statistical Summary")
    st.dataframe(
        get_numeric_summary(df),
        use_container_width=True,
    )

    # Business KPIs
    st.header("4. Business Insights")

    metric_col1, metric_col2 = st.columns(2)

    if "total_sales" in insights:
        metric_col1.metric(
            "Total Sales",
            f"{insights['total_sales']:,.2f}",
        )

    if "total_profit" in insights:
        metric_col2.metric(
            "Total Profit",
            f"{insights['total_profit']:,.2f}",
        )

    # Charts
    st.header("5. Interactive Charts")

    category_col = next(
        (c for c in df.columns if c.strip().lower() == "category"),
        None,
    )
    region_col = next(
        (c for c in df.columns if c.strip().lower() == "region"),
        None,
    )
    sales_col = next(
        (c for c in df.columns if c.strip().lower() == "sales"),
        None,
    )
    profit_col = next(
        (c for c in df.columns if c.strip().lower() == "profit"),
        None,
    )
    date_col = next(
        (c for c in df.columns if c.strip().lower() == "order date"),
        None,
    )

    if category_col and sales_col:
        chart_data = df[[category_col, sales_col]].copy()
        chart_data[sales_col] = pd.to_numeric(
            chart_data[sales_col], errors="coerce"
        )
        chart_data = chart_data.dropna()

        if not chart_data.empty:
            summary = chart_data.groupby(
                category_col, as_index=False
            )[sales_col].sum()

            fig = px.bar(
                summary,
                x=category_col,
                y=sales_col,
                title="Sales by Category",
                labels={sales_col: "Total Sales"},
            )
            st.plotly_chart(fig, use_container_width=True)

    if region_col and sales_col:
        chart_data = df[[region_col, sales_col]].copy()
        chart_data[sales_col] = pd.to_numeric(
            chart_data[sales_col], errors="coerce"
        )
        chart_data = chart_data.dropna()

        if not chart_data.empty:
            summary = chart_data.groupby(
                region_col, as_index=False
            )[sales_col].sum()

            fig = px.bar(
                summary,
                x=region_col,
                y=sales_col,
                title="Sales by Region",
                labels={sales_col: "Total Sales"},
            )
            st.plotly_chart(fig, use_container_width=True)

    if category_col and profit_col:
        chart_data = df[[category_col, profit_col]].copy()
        chart_data[profit_col] = pd.to_numeric(
            chart_data[profit_col], errors="coerce"
        )
        chart_data = chart_data.dropna()

        if not chart_data.empty:
            summary = chart_data.groupby(
                category_col, as_index=False
            )[profit_col].sum()

            fig = px.bar(
                summary,
                x=category_col,
                y=profit_col,
                title="Profit by Category",
                labels={profit_col: "Total Profit"},
            )
            st.plotly_chart(fig, use_container_width=True)

    if date_col and sales_col:
        chart_data = df[[date_col, sales_col]].copy()
        chart_data[date_col] = pd.to_datetime(
            chart_data[date_col], errors="coerce"
        )
        chart_data[sales_col] = pd.to_numeric(
            chart_data[sales_col], errors="coerce"
        )
        chart_data = chart_data.dropna()

        if not chart_data.empty:
            chart_data["Month"] = (
                chart_data[date_col].dt.to_period("M").astype(str)
            )

            monthly = chart_data.groupby(
                "Month", as_index=False
            )[sales_col].sum()

            fig = px.line(
                monthly,
                x="Month",
                y=sales_col,
                markers=True,
                title="Monthly Sales Trend",
            )
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Upload your sample_sales.csv file to begin.")
    

# -------------------------------
# AI DATA ANALYST
# -------------------------------

st.divider()

st.subheader("Ask the AI Data Analyst")

st.write(
    "Ask a question about your dataset, and the AI analyst "
    "will use calculated business insights to answer it."
)

question = st.text_input(
    "Your question",
    placeholder="e.g. Which region has the highest sales?"
)

if st.button("Ask AI Analyst", type="primary"):

    if not question.strip():
        st.warning("Please enter a question first.")

    else:
        with st.spinner("Analyzing your data..."):

            try:
                answer = answer_question(question, df=df)

                st.markdown("### AI Analyst's Answer")
                st.markdown(answer)

            except Exception as error:
                st.error(
                    f"Unable to answer your question: {error}"
                )
    
