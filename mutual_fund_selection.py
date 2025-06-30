# 
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import gspread
from google.oauth2.service_account import Credentials

# --- Styling for Sidebar Multiselect Tags ---
st.markdown("""
<style>
.stSidebar .stMultiSelect [data-baseweb="tag"]:nth-child(1) { background-color: #FFA896 !important; color: white !important; }
.stSidebar .stMultiSelect [data-baseweb="tag"]:nth-child(2) { background-color: #9B1313 !important; color: white !important; }
.stSidebar .stMultiSelect [data-baseweb="tag"]:nth-child(3) { background-color: #38000A !important; color: white !important; }
.stSidebar .stMultiSelect [data-baseweb="tag"]:nth-child(4) { background-color: #DEA193 !important; color: black !important; }
.stSidebar .stMultiSelect [data-baseweb="tag"]:nth-child(5) { background-color: #F88379 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# --- Google Sheets Auth ---
scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
creds_dict = st.secrets["gcp_service_account"]
credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
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

# Load main and mapping sheets
GOOGLE_SHEET_ID = "1hicM1Hs3_7JGcJPTZ9o6iWeoKLOMJXBPJn5gyvjHGw4"
df = load_google_sheet(GOOGLE_SHEET_ID, "Top Schemes")

if df is None or df.empty:
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        df.columns = df.columns.str.strip()
    else:
        st.warning("Please upload an Excel file to proceed.")
        st.stop()

if view == "admin":
    MAPPINGS_SHEET_ID = "1bm-ytBH3qE3JsqOR2x-jMi7-k1DOPKMSPUBRkVhWgkU"
    mappings = load_google_sheet(MAPPINGS_SHEET_ID, "mappings")
    if mappings is None or mappings.empty:
        uploaded_mappings = st.file_uploader("Upload mappings.csv", type=["csv"])
        if uploaded_mappings:
            mappings = pd.read_csv(uploaded_mappings)

# --- Rules and Helpers ---
category_rules = {
    "Not Selected": {"include_category": [], "aum_min": 0, "sharpe_weight": 0, "sortino_weight": 0},
    "Conservative": {"include_category": ["equity: flexi cap", "equity: large & mid cap", "equity: large cap", "equity: index", "equity: dividend yield", "equity: value"], "aum_min": 10000, "sharpe_weight": 0.0, "sortino_weight": 1.0},
    "Moderate Conservative": {"include_category": ["equity: flexi cap", "equity: large & mid cap", "equity: multi cap", "equity: large cap", "equity: index", "equity: dividend yield", "equity: value"], "aum_min": 10000, "sharpe_weight": 0.25, "sortino_weight": 0.75},
    "Moderate": {"include_category": ["equity: flexi cap", "equity: large & mid cap", "equity: multi cap", "equity: large cap", "equity: index", "equity: focused"], "aum_min": 10000, "sharpe_weight": 0.5, "sortino_weight": 0.5},
    "Moderate Aggressive": {"include_category": ["equity: flexi cap", "equity: large & mid cap", "equity: multi cap", "equity: large cap", "equity: mid cap", "equity: focused"], "aum_min": 10000, "sharpe_weight": 0.75, "sortino_weight": 0.25},
    "Aggressive": {"include_category": ["equity: flexi cap", "equity: large & mid cap", "equity: multi cap", "equity: large cap", "equity: mid cap", "equity: focused", "equity: sectoral", "equity: small cap", "equity: thematic"], "aum_min": 10000, "sharpe_weight": 1.0, "sortino_weight": 0.0},
}

# Helper Functions
def filter_funds(df, include_category, aum_min):
    df = df.copy()
    df['AUM(CR)'] = pd.to_numeric(df['AUM(CR)'], errors='coerce')
    return df[df['CATEGORY'].str.strip().str.lower().isin(include_category) & (df['AUM(CR)'] >= aum_min)]

def score_funds(df, sharpe_weight, sortino_weight):
    df = df.copy()
    df['Sharpe_Sortino_Score'] = sharpe_weight * df['SHARPE RATIO'] + sortino_weight * df['SORTINO RATIO']
    return df.sort_values('Sharpe_Sortino_Score', ascending=False)

# --- Sidebar Filters ---
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
top_n = st.sidebar.slider("Number of Top Funds", 5, 50, 10)

# --- Fund Selection ---
df_filtered = filter_funds(df, include_category, aum_min)
df_scored = score_funds(df_filtered, sharpe_weight, sortino_weight)
final_selection = df[df['SCHEMES'].isin(df_scored.head(top_n)['SCHEMES'])].copy()
final_selection = final_selection.merge(df_scored[['SCHEMES', 'Sharpe_Sortino_Score']], on='SCHEMES', how='left')

# --- Column Display Selection ---
default_cols = ['SCHEMES', 'CATEGORY', 'AUM(CR)', 'SHARPE RATIO', 'SORTINO RATIO', 'Sharpe_Sortino_Score']
available_cols = df.columns.tolist()
optional_cols = [col for col in available_cols if col not in default_cols]
extra_cols_selected = st.multiselect("Select additional columns to display:", optional_cols, default=[])
cols_to_display = default_cols + extra_cols_selected

st.subheader("Final Selection")
st.dataframe(final_selection[cols_to_display], use_container_width=True)

# --- Admin: Manual Selection & Overlap ---
if view == "admin":
    if st.sidebar.checkbox("Enable Manual Scheme Analysis"):
        st.sidebar.markdown("### Permanent Scheme Selection")
        default_admin_schemes = [
            "Old Bridge Focused Equity Fund Reg (G)",
            "Parag Parikh Flexi Cap Fund Reg (G)",
            "ICICI Pru Large & Mid Cap Fund Reg (G)"
        ]
        admin_selected_schemes = st.sidebar.multiselect("Select up to 5 schemes for analysis:", df['SCHEMES'].unique().tolist(), default=default_admin_schemes, max_selections=10)

        if admin_selected_schemes:
            st.subheader("Admin Schemes Overview")
            admin_df = df[df['SCHEMES'].isin(admin_selected_schemes)].copy()
            if 'Sharpe_Sortino_Score' in df_scored.columns:
                admin_df = admin_df.merge(df_scored[['SCHEMES', 'Sharpe_Sortino_Score']], on='SCHEMES', how='left')
            st.dataframe(admin_df[cols_to_display], use_container_width=True)

    # Overlap Analysis
    if mappings is not None:
        st.subheader("Overlap Analysis")
        overlap_selection = admin_selected_schemes if 'admin_selected_schemes' in locals() and admin_selected_schemes else st.multiselect("Select schemes for overlap:", df['SCHEMES'].unique(), default=final_selection['SCHEMES'].tolist())
        selected_mapping = mappings[mappings['Investwell'].isin(overlap_selection)]
        uploaded_csvs = st.file_uploader("Upload Holdings CSVs", type="csv", accept_multiple_files=True)

        if st.button("Show Overlap Heatmap"):
            fund_files = [f for f in uploaded_csvs if f.name.replace('.csv', '') in selected_mapping['CSV'].values]
            if fund_files:
                fund_dfs = {f.name: pd.read_csv(f) for f in fund_files}
                unique_stocks = set().union(*(df_f['Invested In'] for df_f in fund_dfs.values()))
                stock_df = pd.DataFrame(0, index=sorted(unique_stocks), columns=[f.name for f in fund_files])
                for f in fund_files:
                    for _, row in fund_dfs[f.name].iterrows():
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