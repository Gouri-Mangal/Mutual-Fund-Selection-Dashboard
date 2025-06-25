import streamlit as st
import pandas as pd
import time
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Mutual Fund Selection Dashboard")

# --- Load Data ---
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


    "Not Selected": {"include_category":[],
        "aum_min": 0,
        "sharpe_weight": 0.0,

        "sortino_weight": 0.0,

    },
    "Conservative": {
        "include_category": ["equity: flexi cap", "equity: large & mid cap", "equity: multi cap", "equity: large cap", "equity: index", "equity: dividend yield", "equity: value", "hybrid: balanced advantage","hybrid: conservative", "hybrid: equity savings", "hybrid: multi-asset"],
        "aum_min": 10000,
        "sharpe_weight": 0.0,
        "sortino_weight": 1.0,
    },
    "Moderate Conservative": {
        "include_category": ["equity: flexi cap", "equity: large & mid cap", "equity: multi cap", "equity: large cap", "equity: index", "equity: dividend yield", "equity: value", "hybrid: balanced advantage","hybrid: conservative", "hybrid: equity savings", "hybrid: multi-asset"],
        "aum_min": 10000,
        "sharpe_weight": 0.25,
        "sortino_weight": 0.75,
    },
    "Moderate": {
        "include_category": ["equity: flexi cap", "equity: large & mid cap", "equity: multi cap", "equity: large cap", "equity: index", "equity: focused"],
        "aum_min": 10000,
        "sharpe_weight": 0.5,
        "sortino_weight": 0.5,
    },
    "Moderate Aggressive": {
        "include_category": ["equity: flexi cap", "equity: large & mid cap", "equity: multi cap", "equity: large cap", "equity: mid cap", "equity: focused"],
        "aum_min": 10000,
        "sharpe_weight": 0.75,
        "sortino_weight": 0.25,
    },
    "Aggressive": {
        "include_category": ["equity: flexi cap", "equity: large & mid cap", "equity: multi cap", "equity: large cap", "equity: mid cap", "equity: focused", "equity: sectoral", "equity: small cap", "equity: thematic"],
        "aum_min": 10000,
        "sharpe_weight": 1.0,
        "sortino_weight": 0.0,
    }
}

mappings = pd.read_csv("mappings.csv")

# --- Helper Functions ---
def filter_funds(df, include_category, aum_min):
    if include_category:
        df_filtered = df[df['CATEGORY'].str.strip().str.lower().isin(include_category)]
    else:
        df_filtered = df.copy()
    df_filtered = df_filtered[df_filtered['AUM(CR)'] >= aum_min]
    df_filtered = pd.concat([df_filtered]).drop_duplicates().reset_index(drop=True)
    return df_filtered.copy()

def score_funds(df, sharpe_weight, sortino_weight):
    df = df.copy()
    df['Sharpe_Sortino_Score'] = (
        sharpe_weight * df['SHARPE RATIO'] +
        sortino_weight * df['SORTINO RATIO']
    )
    return df.sort_values('Sharpe_Sortino_Score', ascending=False)

# --- Sidebar Controls ---
category = st.sidebar.selectbox("Select Risk Category", list(category_rules.keys()))
rules = category_rules[category]
if category == "Not Selected":
    st.warning("Please select a risk category to proceed.")
    st.stop()
# Get lowercase raw categories
all_categories_raw = sorted(df['CATEGORY'].str.strip().str.lower().unique())

# Capitalize for display
all_categories = [cat.title() for cat in all_categories_raw]
default_categories = [cat.title() for cat in rules["include_category"]]

# Multiselect with capitalized options
include_category_display = st.sidebar.multiselect(
    "Manually Select Categories to Include",
    options=all_categories,
    default=default_categories
)

# Convert back to lowercase for filtering logic
include_category = [cat.lower() for cat in include_category_display]

aum_min = st.sidebar.number_input("AUM Minimum (Cr)", value=rules['aum_min'])
sharpe_weight = st.sidebar.slider("Sharpe Weight", 0.0, 1.0, rules['sharpe_weight'])
sortino_weight = st.sidebar.slider("Sortino Weight", 0.0, 1.0, rules['sortino_weight'])
top_n = st.sidebar.slider("Number of Top Funds", min_value=5, max_value=50, value=10)

# --- Filtering and Scoring ---
df_filtered = filter_funds(df, include_category, aum_min)
df_scored = score_funds(df_filtered, sharpe_weight, sortino_weight)

# --- Personal Selection Widget ---
default_selection = list(df_scored.head(top_n)['SCHEMES'])
manual_selection = st.multiselect(
    "Personal Selection: Add or Remove Schemes",
    options=list(df['SCHEMES']),
    default=default_selection
)
final_selection = df[df['SCHEMES'].isin(manual_selection)].reset_index(drop=True)
# Merge Sharpe_Sortino_Score from df_scored (if available)
final_selection = final_selection.merge(
    df_scored[['SCHEMES', 'Sharpe_Sortino_Score']],
    on='SCHEMES',
    how='left'
).sort_values('Sharpe_Sortino_Score', ascending=False)
st.subheader("Final Selection")
st.dataframe(final_selection[['SCHEMES', 'CATEGORY', 'AUM(CR)', 'SHARPE RATIO', 'SORTINO RATIO', 'Sharpe_Sortino_Score']], hide_index=True, use_container_width=True)

# --- Instructions for CSVs ---
fundata_folder = f"fundata_{category.lower().replace(' ', '_')}"
st.markdown(f"** Download the holdings CSVs for these schemes and place them in the folder: `{fundata_folder}` (create it if it doesn't exist).")
st.write(final_selection['SCHEMES'].tolist())

# # --- Manual Checklist with Checkboxes ---
# st.markdown("#### Manual Checklist: Mark files as downloaded and placed in folder")

# checkbox_states = {}
# for scheme in final_selection['SCHEMES']:
#     checkbox_states[scheme] = st.checkbox(f"{scheme}")

# checked_count = sum(checkbox_states.values())
# total_count = len(checkbox_states)

# st.info(f"{checked_count} of {total_count} files checked as present.")

# if checked_count == total_count:
#     st.success("All files checked! You can proceed with overlap analysis.")
# else:
#     st.warning("Some files are not checked. Please ensure all files are downloaded and placed in the folder before proceeding.")

# # --- Manual Checklist with Checkboxes ---
# st.markdown("### Checklist: Select files to include in overlap analysis")

# checkbox_states = {}
# for scheme in final_selection['SCHEMES']:
#     expected_file = scheme_to_filename(scheme)
#     checkbox_states[scheme] = st.checkbox(f"{scheme}  (expected file: {expected_file})", value=True)

# checked_schemes = [scheme for scheme, checked in checkbox_states.items() if checked]
# checked_count = len(checked_schemes)
# total_count = len(checkbox_states)

# st.info(f"{checked_count} of {total_count} files selected for analysis.")

# if st.button("Show Overlap Heatmap (for selected files)"):
#     selected_files = [scheme_to_filename(s) for s in checked_schemes]
#     present_files = [f for f in selected_files if os.path.exists(os.path.join(fundata_folder, f))]
#     if not present_files:
#         st.warning(f"No matching CSV files found in '{fundata_folder}'. Please check your selection and file names.")
#     else:
#         unique_stocks = set()
#         for file_name in present_files:
#             df_scheme = pd.read_csv(os.path.join(fundata_folder, file_name))
#             unique_stocks.update(df_scheme['Invested In'])
#         stock_df = pd.DataFrame(0, index=sorted(unique_stocks), columns=present_files)
#         for file_name in present_files:
#             df_scheme = pd.read_csv(os.path.join(fundata_folder, file_name))
#             for _, row in df_scheme.iterrows():
#                 stock = row['Invested In']
#                 percent = row['% of Total Holding']
#                 stock_df.at[stock, file_name] = percent
#         corr_mf = stock_df.corr()
#         fig, ax = plt.subplots(figsize=(8, 6))
#         sns.heatmap(corr_mf, annot=True, cmap='YlGnBu', fmt='.2f', ax=ax)
#         st.pyplot(fig)
#         row_sum_abs = corr_mf.abs().sum(axis=1).sort_values(ascending=True)
#         st.write("Sum of absolute correlations for each selected fund:")
#         st.write(row_sum_abs)

# --- Load Data ---
fund_files = st.file_uploader("Upload your Excel file", type=["csv"], accept_multiple_files=True)

# --- Map SCHEME names to CSV filenames using mapping DataFrame ---
final_selection_with_csv = final_selection.merge(
    mappings[['Investwell', 'CSV']],
    left_on='SCHEMES',
    right_on='Investwell',
    how='left'
)

# Build list of file paths for available CSVs



# --- Overlap Analysis Button ---
if st.button("Show Overlap Heatmap (after placing CSVs)"):
    if fund_files:
        # Build a mapping: filename → DataFrame
        fund_dfs = {}
        unique_stocks = set()
        for uploaded_file in fund_files:



            df_scheme = pd.read_csv(uploaded_file)
            fund_dfs[uploaded_file.name] = df_scheme
            unique_stocks.update(df_scheme['Invested In'])

        # Build the stock_df DataFrame
        stock_df = pd.DataFrame(0, index=sorted(unique_stocks), columns=[f.name for f in fund_files])
        for uploaded_file in fund_files:
            df_scheme = fund_dfs[uploaded_file.name]
            for _, row in df_scheme.iterrows():
                stock = row['Invested In']
                percent = row['% of Total Holding']
                stock_df.at[stock, uploaded_file.name] = percent

        # Now you can do your overlap analysis as before
        corr_mf = stock_df.corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_mf, annot=True, cmap='YlGnBu', fmt='.2f', ax=ax, annot_kws={"size": 8})
        st.pyplot(fig)
        row_sum_abs = corr_mf.abs().sum(axis=1).sort_values(ascending=True)
        st.write("Sum of absolute correlations for each selected fund:")
        st.write(row_sum_abs)
    else:
        st.info("Please upload your holdings CSV files above to run overlap analysis.")


# # --- Custom Overlap Analysis for Selected Schemes ---
# st.markdown("### Optional: Select Schemes for Custom Overlap Analysis")

# custom_selected = st.multiselect(
#     "Pick schemes from your final selection for a custom overlap heatmap:",
#     options=list(final_selection['SCHEMES']),
#     default=list(final_selection['SCHEMES'])
# )

# if st.button("Show Custom Overlap Heatmap"):
#     # Build expected file names for selected schemes
#     present_file_names = [f for f in os.listdir(fundata_folder) if f.endswith('.csv')]
#     if not present_file_names:
#         st.warning(f"No CSV files found in '{fundata_folder}'. Please add the files and try again.")
#     else:
#         unique_stocks = set()
#         for present_name in present_files_names:
#             df_scheme = pd.read_csv(os.path.join(fundata_folder, present_name))
#             unique_stocks.update(df_scheme['Invested In'])
#         stock_df = pd.DataFrame(0, index=sorted(unique_stocks), columns=present_files_names)
#         for present_name in present_files_names:
#             df_scheme = pd.read_csv(os.path.join(fundata_folder, present_name))
#             for _, row in df_scheme.iterrows():
#                 stock = row['Invested In']
#                 percent = row['% of Total Holding']
#                 stock_df.at[stock, file_name] = percent
#         corr_mf = stock_df.corr()
#         fig, ax = plt.subplots(figsize=(8, 6))
#         sns.heatmap(corr_mf, annot=True, cmap='YlGnBu', fmt='.2f', ax=ax)
#         st.pyplot(fig)
#         row_sum_abs = corr_mf.abs().sum(axis=1).sort_values(ascending=True)
#         st.write("Sum of absolute correlations for each selected fund:")
#         st.write(row_sum_abs)

