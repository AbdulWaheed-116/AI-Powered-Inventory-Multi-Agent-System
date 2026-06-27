import os
import pandas as pd
import streamlit as st
from agents.data_agent import DataAgent

st.title("Agentic Inventory Optimization System")

uploaded_file = st.file_uploader("Upload Inventory CSV", type=["csv"])

if uploaded_file is not None:
    st.success("File uploaded successfully")

    # 1. Save uploaded file to data/raw/
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    raw_file_path = os.path.join(raw_dir, uploaded_file.name)

    # Save the file bytes
    with open(raw_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 2. Call DataAgent
    agent = DataAgent()
    df = agent.read_csv(raw_file_path)

    # 3. Clean CSV
    missing_report = agent.check_missing_values()
    duplicates_removed, df_cleaned = agent.remove_duplicates()

    # 4. Save result to data/processed/cleaned_data.csv
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    cleaned_file_path = os.path.join(processed_dir, "cleaned_data.csv")
    agent.save_cleaned_csv(cleaned_file_path)

    # 5. Display cleaning summary
    st.subheader("Data Cleaning Summary")

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Original Row Count", len(df_cleaned) + duplicates_removed)
    col2.metric("Duplicates Removed", duplicates_removed)
    col3.metric("Cleaned Row Count", len(df_cleaned))

    # Missing values report
    st.write("### Missing Values Analysis")
    if missing_report["has_missing"]:
        st.warning(f"Found {missing_report['total_missing']} missing values in the dataset.")
        # Filter columns that have missing values for cleaner presentation
        cols_with_missing = {
            col: count for col, count in missing_report["missing_by_column"].items() if count > 0
        }
        missing_df = pd.DataFrame(
            list(cols_with_missing.items()),
            columns=["Column Name", "Missing Values Count"]
        )
        st.dataframe(missing_df, use_container_width=True)
    else:
        st.info("No missing values detected.")

    # Dataset Preview
    st.write("### Cleaned Dataset Preview")
    st.dataframe(df_cleaned.head(10), use_container_width=True)