
import streamlit as st
import pandas as pd
import numpy as np
import re
import json
from google import genai

# ---------------------------------
# 1. WEBSITE CONFIGURATION
# ---------------------------------

st.set_page_config(
    page_title="FinSight AI",
    page_icon="📊",
    layout="wide"
)

st.title("📊 FinSight AI")
st.caption("Your AI-powered financial analysis platform")

st.write(
    "Upload a company's financial statement to calculate "
    "financial ratios, compare performance and identify "
    "potential financial warning signs."
)

# ---------------------------------
# 2. FINANCIAL COLUMN RECOGNITION
# ---------------------------------

ALIASES = {
    "Year": [
        "year", "financial_year", "fiscal_year", "fy"
    ],
    "Revenue": [
        "revenue", "sales", "total_sales",
        "total_revenue", "total_income", "operating_revenue"
    ],
    "COGS": [
        "cogs", "cost_of_goods_sold",
        "cost_of_sales", "cost_of_revenue"
    ],
    "Operating_Expenses": [
        "operating_expenses", "operating_costs",
        "opex", "total_operating_expenses"
    ],
    "Interest_Expense": [
        "interest_expense", "finance_cost",
        "finance_costs", "interest_cost"
    ],
    "Tax_Expense": [
        "tax_expense", "income_tax", "tax", "taxes"
    ],
    "Total_Assets": [
        "total_assets", "assets", "total_asset"
    ],
    "Current_Assets": [
        "current_assets", "total_current_assets"
    ],
    "Inventory": [
        "inventory", "stock", "inventories"
    ],
    "Current_Liabilities": [
        "current_liabilities", "total_current_liabilities"
    ],
    "Total_Debt": [
        "total_debt", "debt", "borrowings",
        "total_borrowings", "total_loans"
    ],
    "Shareholders_Equity": [
        "shareholders_equity", "equity", "net_worth",
        "owners_equity", "total_equity", "shareholders_funds"
    ],
    "Net_Income": [
        "net_income", "net_profit", "profit_after_tax",
        "pat", "net_profit_after_tax", "profit"
    ],
    "Operating_Cash_Flow": [
        "operating_cash_flow", "cash_flow_from_operations",
        "cash_from_operations",
        "net_cash_from_operating_activities"
    ]
}


def clean_name(name):
    name = str(name).lower().strip()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def standardize_financial_data(raw_df):

    data = raw_df.copy()

    alias_lookup = {}

    for standard, names in ALIASES.items():
        for name in names:
            alias_lookup[clean_name(name)] = standard

    rename_map = {}

    for column in data.columns:
        cleaned = clean_name(column)

        if cleaned in alias_lookup:
            rename_map[column] = alias_lookup[cleaned]

    data = data.rename(columns=rename_map)

    duplicates = data.columns[data.columns.duplicated()].tolist()

    if duplicates:
        raise ValueError(
            "Multiple columns represent the same financial field: "
            + str(duplicates)
        )

    if "Year" not in data.columns:
        raise ValueError("A Year or Financial Year column is required.")

    if "Revenue" not in data.columns:
        raise ValueError("A Revenue or Sales column is required.")

    for column in data.columns:
        if column != "Year":
            values = (
                data[column]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("₹", "", regex=False)
                .str.replace("Rs.", "", regex=False)
                .str.replace("Rs", "", regex=False)
                .str.strip()
            )
            data[column] = pd.to_numeric(values, errors="coerce")

    data["Year"] = data["Year"].astype(str).str.strip()

    return data


# ---------------------------------
# 3. FINANCIAL RATIO CALCULATIONS
# ---------------------------------

def calculate_ratios(data):

    results = []

    for _, row in data.iterrows():

        result = {"Year": row["Year"]}

        current_assets = row.get("Current_Assets", np.nan)
        current_liabilities = row.get("Current_Liabilities", np.nan)
        inventory = row.get("Inventory", np.nan)
        debt = row.get("Total_Debt", np.nan)
        equity = row.get("Shareholders_Equity", np.nan)
        profit = row.get("Net_Income", np.nan)
        revenue = row.get("Revenue", np.nan)
        assets = row.get("Total_Assets", np.nan)

        if pd.notna(current_assets) and pd.notna(current_liabilities) and current_liabilities > 0:
            result["Current Ratio"] = round(
                current_assets / current_liabilities, 2
            )

        if (
            pd.notna(current_assets)
            and pd.notna(inventory)
            and pd.notna(current_liabilities)
            and current_liabilities > 0
        ):
            result["Quick Ratio"] = round(
                (current_assets - inventory) / current_liabilities, 2
            )

        if pd.notna(debt) and pd.notna(equity) and equity > 0:
            result["Debt to Equity"] = round(debt / equity, 2)

        if pd.notna(profit) and pd.notna(revenue) and revenue != 0:
            result["Net Profit Margin (%)"] = round(
                profit / revenue * 100, 2
            )

        if pd.notna(profit) and pd.notna(equity) and equity > 0:
            result["ROE (%)"] = round(profit / equity * 100, 2)

        if pd.notna(revenue) and pd.notna(assets) and assets > 0:
            result["Asset Turnover"] = round(revenue / assets, 2)

        results.append(result)

    return pd.DataFrame(results)


# ---------------------------------
# 4. YEAR-ON-YEAR COMPARISON
# ---------------------------------

def compare_performance(data):

    data = data.copy()

    if len(data) < 2:
        return {
            "Message": "At least two financial years are needed "
                       "for growth comparison."
        }

    # Sort by year label
    data = data.sort_values("Year")

    old = data.iloc[0]
    new = data.iloc[-1]

    comparison = {
        "Starting year": old["Year"],
        "Latest year": new["Year"]
    }

    for field in [
        "Revenue", "Net_Income",
        "Total_Assets", "Operating_Cash_Flow"
    ]:

        if field not in data.columns:
            continue

        old_value = old[field]
        new_value = new[field]

        if pd.isna(old_value) or pd.isna(new_value):
            comparison[field + " Growth (%)"] = "Insufficient data"

        elif old_value == 0:
            comparison[field + " Growth (%)"] = (
                "Cannot calculate: starting value is zero"
            )

        else:
            growth = ((new_value - old_value) / abs(old_value)) * 100
            comparison[field + " Growth (%)"] = round(growth, 2)

    return comparison


# ---------------------------------
# 5. PRELIMINARY RISK SCREENING
# ---------------------------------

def identify_risks(data):

    risks = []

    for _, row in data.iterrows():

        year = row["Year"]

        ca = row.get("Current_Assets", np.nan)
        cl = row.get("Current_Liabilities", np.nan)
        debt = row.get("Total_Debt", np.nan)
        equity = row.get("Shareholders_Equity", np.nan)
        profit = row.get("Net_Income", np.nan)
        revenue = row.get("Revenue", np.nan)
        cash = row.get("Operating_Cash_Flow", np.nan)

        if pd.notna(ca) and pd.notna(cl) and cl > 0:
            if ca / cl < 1:
                risks.append(
                    f"{year}: Current ratio is below 1; "
                    "possible short-term liquidity concern."
                )

        if pd.notna(debt) and pd.notna(equity) and equity > 0:
            if debt / equity > 2:
                risks.append(
                    f"{year}: Debt-to-equity ratio is above 2; "
                    "review borrowing levels."
                )

        if pd.notna(profit) and pd.notna(revenue) and revenue > 0:
            if profit / revenue * 100 < 5:
                risks.append(
                    f"{year}: Net profit margin is below 5%; "
                    "review profitability."
                )

        if pd.notna(cash) and pd.notna(profit):
            if cash < profit:
                risks.append(
                    f"{year}: Operating cash flow is below net income; "
                    "review cash conversion."
                )

    if not risks:
        risks.append(
            "No risks detected by the selected preliminary "
            "screening rules. This does not guarantee financial health."
        )

    return risks


# ---------------------------------
# 6. FILE UPLOAD
# ---------------------------------

st.header("1. Upload financial statements")

uploaded_file = st.file_uploader(
    "Choose an Excel or CSV file",
    type=["xlsx", "xls", "csv"]
)

if uploaded_file is not None:

    try:

        if uploaded_file.name.lower().endswith(".csv"):
            raw_data = pd.read_csv(uploaded_file)
        else:
            raw_data = pd.read_excel(uploaded_file)

        company_name = st.text_input(
            "Company name (optional)",
            value="Uploaded Company"
        )

        data = standardize_financial_data(raw_data)

        st.success("Financial file uploaded and standardised!")

        st.subheader("2. Uploaded financial data")
        st.dataframe(data, use_container_width=True)

        st.subheader("3. Financial ratios")

        ratios = calculate_ratios(data)

        st.dataframe(ratios, use_container_width=True)

        st.subheader("4. Performance comparison")

        comparison = compare_performance(data)

        st.json(comparison)

        st.subheader("5. Potential financial risks")

        risks = identify_risks(data)

        for risk in risks:
            st.write("•", risk)

        # ---------------------------------
        # 7. FINANCIAL CHART
        # ---------------------------------

        st.subheader("6. Financial performance chart")

        chart_fields = [
            field for field in ["Revenue", "Net_Income"]
            if field in data.columns
        ]

        if chart_fields:
            chart_data = data.set_index("Year")[chart_fields]
            st.bar_chart(chart_data)

        # ---------------------------------
        # 8. GEMINI AI REPORT
        # ---------------------------------

        st.subheader("7. FinSight AI financial report")

        if st.button("Generate AI Financial Report"):

            try:
                api_key = st.secrets["GEMINI_API_KEY"]

                client = genai.Client(api_key=api_key)

                analysis_data = {
                    "company": company_name,
                    "financial_data": data.to_dict(orient="records"),
                    "ratios": ratios.to_dict(orient="records"),
                    "performance_comparison": comparison,
                    "risk_screening": risks
                }

                prompt = f"""
                You are FinSight AI, a careful financial analyst.

                Analyse this company's financial information:

                {json.dumps(analysis_data, default=str)}

                Explain:
                1. Revenue and profit performance.
                2. Liquidity and solvency, where data is available.
                3. Profitability and efficiency, where data is available.
                4. Potential financial risks.
                5. Overall financial observations.

                Use simple language.
                Do not invent missing information.
                Distinguish calculated facts from interpretations.
                Do not claim that the company is risk-free.
                Mention that this is an automated preliminary analysis,
                not a formal audit or investment recommendation.
                """

                with st.spinner("FinSight AI is analysing the company..."):

                    response = client.models.generate_content(
                        model="gemini-3.5-flash-lite",
                        contents=prompt
                    )

                st.markdown(response.text)

            except Exception as error:
                st.error(
                    "AI report could not be generated. "
                    "Check your Gemini API key and deployment settings."
                )
                st.caption(str(error))

    except Exception as error:
        st.error("Unable to process this financial file.")
        st.write(str(error))

st.divider()

st.caption(
    "FinSight AI | Automated financial analysis. "
    "Results depend on the accuracy and completeness of uploaded data."
)
