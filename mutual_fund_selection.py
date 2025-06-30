import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import gspread
from google.oauth2.service_account import Credentials

st.markdown("""
<style>
.stSidebar .stMultiSelect [data-baseweb="tag"]:nth-child(1) {
    background-color: #FFA896 !important; color: white !important;
}
.stSidebar .stMultiSelect [data-baseweb="tag"]:nth-child(2) {
    background-color: #9B1313 !important; color: white !important;
}
.stSidebar .stMultiSelect [data-baseweb="tag"]:nth-child(3) {
    background-color: #38000A !important; color: white !important;
}
.stSidebar .stMultiSelect [data-baseweb="tag"]:nth-child(4) {
    background-color: #DEA193 !important; color: black !important;
}
.stSidebar .stMultiSelect [data-baseweb="tag"]:nth-child(5) {
    background-color: #F88379 !important; color: white !important;
}
.stSidebar .stMultiSelect [data-baseweb="tag"]:nth-child(6) {
    background-color: #E26F66 !important; color: white !important;
}
.stSidebar .stMultiSelect [data-baseweb="tag"]:nth-child(7) {
    background-color: #BE5103 !important; color: white !important;
}
.stSidebar .stMultiSelect [data-baseweb="tag"]:nth-child(8) {
    background-color: #9E3A26 !important; color: white !important;
}
.stSidebar .stMultiSelect [data-baseweb="tag"]:nth-child(9) {
    background-color: #DA2C43 !important; color: white !important;
}
.stSidebar .stMultiSelect [data-baseweb="tag"]:nth-child(10) {
    background-color: #FA5053 !important; color: white !important;
}
</style>
""", unsafe_allow_html=True)


# Define the scopes
scopes = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# Load credentials from Streamlit Secrets
creds_dict = st.secrets["gcp_service_account"]
credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)

# Authorize gspread with those credentials
gc = gspread.authorize(credentials)

# --- View Detection ---
query_params = st.query_params
view = query_params.get("view")
if view not in ["user", "admin"]:
    st.query_params.update({"view": "user"})
    st.stop()

# --- Load from Google Sheets ---
@st.cache_data(ttl=600)
def load_google_sheet(sheet_id, sheet_name):
    try:
        sh = gc.open_by_key(sheet_id)
        worksheet = sh.worksheet(sheet_name)
        data = worksheet.get_all_records()
        return pd.DataFrame(data)
    except Exception:
        return None

# --- Load Main Sheet ---
GOOGLE_SHEET_ID = "1hicM1Hs3_7JGcJPTZ9o6iWeoKLOMJXBPJn5gyvjHGw4"
SHEET_NAME = "Top Schemes"
df = load_google_sheet(GOOGLE_SHEET_ID, SHEET_NAME)

if df is not None and not df.empty:
    st.success("Loaded data from Google Sheet!")
else:
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        df.columns = df.columns.str.strip()
        st.success("File uploaded and loaded successfully!")
    else:
        st.warning("Please upload an Excel file to proceed.")
        st.stop()

# --- Category Rules ---
category_rules = {
    "Not Selected": {

        "include_category":[],
        "aum_min": 0, 
        "sharpe_weight": 0,
        "sortino_weight": 0

        },

    "Conservative": {

        "include_category": ["equity: flexi cap", "equity: large & mid cap", "equity: large cap", "equity: index", "equity: dividend yield", "equity: value"], 
        "aum_min": 10000, 
        "sharpe_weight": 0.0, 
        "sortino_weight": 1.0
        
        },

    "Moderate Conservative": {

        "include_category": ["equity: flexi cap", "equity: large & mid cap", "equity: multi cap", "equity: large cap", "equity: index", "equity: dividend yield", "equity: value"], 
        "aum_min": 10000,
        "sharpe_weight": 0.25,
        "sortino_weight": 0.75
        
        },

    "Moderate": {

        "include_category": ["equity: flexi cap", "equity: large & mid cap", "equity: multi cap", "equity: large cap", "equity: index", "equity: focused"], 
        "aum_min": 10000, 
        "sharpe_weight": 0.5, 
        "sortino_weight": 0.5
        
        },

    "Moderate Aggressive": {
        
        "include_category": ["equity: flexi cap", "equity: large & mid cap", "equity: multi cap", "equity: large cap", "equity: mid cap", "equity: focused"],
        "aum_min": 10000,
        "sharpe_weight": 0.75,
        "sortino_weight": 0.25
        
        },

    "Aggressive": {
        
        "include_category": ["equity: flexi cap", "equity: large & mid cap", "equity: multi cap", "equity: large cap", "equity: mid cap", "equity: focused", "equity: sectoral", "equity: small cap", "equity: thematic"], 
        "aum_min": 10000,
        "sharpe_weight": 1.0,
        "sortino_weight": 0.0
        
        }
}

# --- Load Mappings Sheet ---
mappings = None
if view == "admin":
    MAPPINGS_SHEET_ID = "1bm-ytBH3qE3JsqOR2x-jMi7-k1DOPKMSPUBRkVhWgkU"
    MAPPINGS_SHEET_NAME = "mappings"
    mappings = load_google_sheet(MAPPINGS_SHEET_ID, MAPPINGS_SHEET_NAME)
    if mappings is not None and not mappings.empty:
        st.success("Loaded Mappings from Google Sheet!")
    else:
        uploaded_mappings = st.file_uploader("Upload mappings.csv", type=["csv"])
        if uploaded_mappings is not None:
            mappings = pd.read_csv(uploaded_mappings)

# --- Helper Functions ---
def filter_funds(df, include_category, aum_min):
    df = df.copy()
    df['AUM(CR)'] = pd.to_numeric(df['AUM(CR)'], errors='coerce')
    if include_category:
        df_filtered = df[df['CATEGORY'].str.strip().str.lower().isin(include_category)]
    else:
        df_filtered = df.copy()
    return df_filtered[df_filtered['AUM(CR)'] >= aum_min].drop_duplicates().reset_index(drop=True)

def score_funds(df, sharpe_weight, sortino_weight):
    df = df.copy()
    df['Sharpe_Sortino_Score'] = sharpe_weight * df['SHARPE RATIO'] + sortino_weight * df['SORTINO RATIO']
    return df.sort_values('Sharpe_Sortino_Score', ascending=False)

# --- Sidebar Controls ---
category = st.sidebar.selectbox("Select Risk Category", list(category_rules.keys()))
rules = category_rules[category]
if category == "Not Selected":
    st.warning("Please select a risk category to proceed.")
    st.stop()

all_categories_raw = sorted(df['CATEGORY'].str.strip().str.lower().unique())
all_categories = [cat.title() for cat in all_categories_raw]
default_categories = [cat.title() for cat in rules["include_category"]]

include_category_display = st.sidebar.multiselect("Manually Select Categories to Include", options=all_categories, default=default_categories)
include_category = [cat.lower() for cat in include_category_display]

aum_min = st.sidebar.number_input("AUM Minimum (Cr)", value=rules['aum_min'])
sharpe_weight = st.sidebar.slider("Sharpe Weight", 0.0, 1.0, rules['sharpe_weight'])
sortino_weight = st.sidebar.slider("Sortino Weight", 0.0, 1.0, rules['sortino_weight'])
top_n = st.sidebar.slider("Number of Top Funds", min_value=5, max_value=50, value=10)

if view == "admin":
    st.sidebar.markdown("### Admin: Permanent Scheme Selection")

    # Predefined default schemes (make sure names match exactly what's in df['SCHEMES'])
    default_admin_schemes = [
        "Old Bridge Focused Equity Fund Reg (G)", "Parag Parikh Flexi Cap Fund Reg (G)",
        "ICICI Pru Large & Mid Cap Fund Reg (G)"
    ]

    admin_selected_schemes = st.sidebar.multiselect(
        "Select up to 5 schemes for analysis:",
        options=list(df['SCHEMES'].unique()),
        default=[s for s in default_admin_schemes if s in df['SCHEMES'].unique()],
        max_selections=10
    )

    if admin_selected_schemes:
        st.subheader("Permanent Analysis: Selected Schemes Overview")
        admin_df = df[df['SCHEMES'].isin(admin_selected_schemes)].copy()

        # Merge score if available
        if 'Sharpe_Sortino_Score' in df_scored.columns:
            admin_df = admin_df.merge(df_scored[['SCHEMES', 'Sharpe_Sortino_Score']], on='SCHEMES', how='left')

        # Columns to show
        analysis_cols = ['SCHEMES', 'CATEGORY', 'AUM(CR)', 'EXPENSE RATIO', 'SHARPE RATIO',
                         'SORTINO RATIO', 'FUND RATING', 'ALPHA', 'BETA', 'STANDARD DEV',
                         'Sharpe_Sortino_Score']
        analysis_cols = [col for col in analysis_cols if col in admin_df.columns]  # in case of missing

        st.dataframe(admin_df[analysis_cols], hide_index=True, use_container_width=True)

# --- Filtering, Scoring, and Final Selection ---
df_filtered = filter_funds(df, include_category, aum_min)
df_scored = score_funds(df_filtered, sharpe_weight, sortino_weight)
default_selection = list(df_scored.head(top_n)['SCHEMES'])
manual_selection = st.multiselect("Personal Selection: Add or Remove Schemes", options=list(df['SCHEMES']), default=default_selection)

final_selection = df[df['SCHEMES'].isin(manual_selection)].reset_index(drop=True)
final_selection = final_selection.merge(df_scored[['SCHEMES', 'Sharpe_Sortino_Score']], on='SCHEMES', how='left').sort_values('Sharpe_Sortino_Score', ascending=False)
final_selection.columns = final_selection.columns.str.strip()
# --- Final Selection Display with Custom Columns ---
default_cols = ['SCHEMES', 'CATEGORY', 'AUM(CR)', 'SHARPE RATIO', 'SORTINO RATIO', 'Sharpe_Sortino_Score']
available_cols = ['SCHEMES', 'EXPENSE RATIO', 'CATEGORY', 'AUM(CR)', '1 DAY', '7 DAY', '15 DAY', '30 DAY', '3 MONTH',
                  '6 MONTH', '1 YEAR', '2 YEAR', '3 YEAR', '5 YEAR', '7 YEAR', '10 YEAR', '15 YEAR', '20 YEAR',
                  '25 YEAR', 'SINCE INCEPTION RETURN', 'FUND RATING', 'ALPHA', 'BETA', 'MEAN', 'STANDARD DEV',
                  'SHARPE RATIO', 'SORTINO RATIO', 'AVERAGE MATURITY', 'MODIFIED DURATION', 'YIELD TO MATURITY',
                  'LAUNCH DATE', 'SCHEME BENCHMARK', 'LARGECAP RATIO', 'MIDCAP RATIO', 'SMALLCAP RATIO',
                  'CURRENT NAV', 'IS RECOMMENDED']

# Exclude default columns to create optional ones
optional_cols = [col for col in available_cols if col not in default_cols]

extra_cols_selected = st.multiselect(
 "Select additional columns to display:",
    options=optional_cols,
    default=[]
)

cols_to_display = default_cols + extra_cols_selected

st.subheader("Final Selection")
st.dataframe(final_selection[cols_to_display], hide_index=True, use_container_width=True)

# st.subheader("Final Selection")
# st.dataframe(final_selection[['SCHEMES', 'CATEGORY', 'AUM(CR)', 'SHARPE RATIO', 'SORTINO RATIO', 'Sharpe_Sortino_Score']], hide_index=True, use_container_width=True)

# --- Admin Section: Overlap Analysis ---
if view == "admin" and mappings is not None:
    st.write("### Overlap Analysis")
    if admin_selected_schemes:
     overlap_selection = admin_selected_schemes
     st.info("Using sidebar selection for overlap analysis.")
    else:
     overlap_selection = st.multiselect("Select schemes for overlap analysis:", options=list(df['SCHEMES']), default=final_selection['SCHEMES'].tolist())
    
    selected_mapping = mappings[mappings['Investwell'].isin(overlap_selection)]
    uploaded_csvs = st.file_uploader("Upload Holdings CSVs for Overlap Analysis", type=["csv"], accept_multiple_files=True)

    required_csv_names = selected_mapping['CSV'].dropna().unique()
    fund_files = [f for f in uploaded_csvs if f.name.replace('.csv', '') in required_csv_names]

    if st.button("Show Overlap Heatmap"):
        if fund_files:
            fund_dfs = {}
            unique_stocks = set()
            for f in fund_files:
                df_f = pd.read_csv(f)
                fund_dfs[f.name] = df_f
                unique_stocks.update(df_f['Invested In'])

            stock_df = pd.DataFrame(0, index=sorted(unique_stocks), columns=[f.name for f in fund_files])
            for f in fund_files:
                df_f = fund_dfs[f.name]
                for _, row in df_f.iterrows():
                    stock_df.at[row['Invested In'], f.name] = row['% of Total Holding']

            corr_mf = stock_df.corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_mf, annot=True, cmap='YlGnBu', fmt='.2f', ax=ax, annot_kws={"size": 12}, linecolor="black", linewidths=0.5)
            st.pyplot(fig)

            row_sum_abs = corr_mf.abs().sum(axis=1).sort_values(ascending=True)
            st.write("Sum of absolute correlations for each selected fund:")
            st.write(row_sum_abs)

            csv_data = stock_df.reset_index().rename(columns={"index": "Stock"}).to_csv(index=False)
            st.download_button("Download Stock-wise Holdings CSV", data=csv_data, file_name="stock_holdings_overlap.csv", mime="text/csv")
        else:
            st.info("Please upload your holdings CSV files above to run overlap analysis.")
