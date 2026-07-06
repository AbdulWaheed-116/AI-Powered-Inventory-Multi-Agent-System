
import os
import json
import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import requests
import logging
import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("GeminiAssistant")

# Load environment variables from .env
load_dotenv()

# Import the existing agents
from agents.host_agent.agent import HostAgent
from agents.optimization_agent.agent import OptimizationAgent
from agents.reporting_agent.agent import ReportingAgent

def load_dotenv_key():
    """
    Retrieve GEMINI_API_KEY from environment variables (loaded via python-dotenv)
    """
    load_dotenv()
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key.strip()
    return None

def call_gemini(api_key, prompt):
    """
    Calls the Gemini API using the official Google GenAI SDK.
    Handles invalid API keys, 404 model not found, 429 quota exceeded, network timeouts, and empty responses.
    """
   

    # Validate that the API key exists before creating the client
    if not api_key:
        logger.error("Gemini API Key validation failed: API key is missing or empty.")
        return "🔑 **Gemini API Key missing!** Please add your API key as a `GEMINI_API_KEY` environment variable or create a `.env` file in the project root containing `GEMINI_API_KEY=\"your_api_key_here\"`."
    
    try:
        # Create client with timeout configured via http_options (30 seconds)
        client = genai.Client(api_key=api_key, http_options={'timeout': 30000})
        print("Calling Gemini...")
        print("Prompt length:", len(prompt))
        print("prompt",prompt)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        
        # Handle empty responses
        if not response or not response.text or not response.text.strip():
            logger.warning("Gemini API call succeeded but returned an empty response.")
            return "⚠️ **Empty Response**: The AI assistant returned an empty response. Please try rephrasing your question."
            
        return response.text
        
    except errors.APIError as e:
        logger.error(f"Google GenAI SDK API Error: Code={e.code}, Message={e.message}, Status={e.status}", exc_info=True)
        if e.code == 429:
            return "⚠️ **Quota Exceeded (429 Error)**: You have hit the rate limit for the Gemini API. Please wait a moment and try again."
        elif e.code == 404:
            return "❌ **Not Found (404 Error)**: The requested Gemini model or resource was not found. Please verify the model configuration."
        elif e.code == 400 and ("API key not valid" in (e.message or "") or "API_KEY_INVALID" in str(e)):
            return "🔑 **Invalid API Key**: The provided Gemini API key is invalid. Please check your `.env` file or environment variables."
        else:
            return f"❌ **Gemini API Error ({e.code})**: {e.message or str(e)}"
            
    except httpx.TimeoutException as e:
        logger.error("Network timeout occurred during Gemini API call", exc_info=True)
        return "⚠️ **Network Timeout**: The connection to the Gemini API timed out. Please check your internet connection or try again later."
        
    except Exception as e:
        logger.error(f"Unexpected exception during Gemini API call: {str(e)}", exc_info=True)
        return f"❌ **Unexpected Error**: {str(e)}"

import asyncio
import concurrent.futures

def run_async_in_thread(coro):
    """
    Executes an asynchronous coroutine in a separate worker thread.
    This prevents 'This event loop is already running' errors in Streamlit.
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()

async def call_adk_root_agent(prompt: str, session_id: str, raw_data_path: str = None) -> str:
    """
    Calls the ADK Root Agent using the official InMemoryRunner API.
    Collects and returns all generated response text chunks.
    """
    from google.adk.runners import InMemoryRunner
    from google.genai import types
    from adk.root_agent import root_agent
    
    runner = InMemoryRunner(agent=root_agent)
    runner.auto_create_session = True
    
    user_id = "streamlit_user"
    # Pre-create session to avoid SessionNotFoundError
    session = await runner.session_service.get_session(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session_id
    )
    if not session:
        await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id=user_id,
            session_id=session_id
        )
        
    # Construct UserContent as expected by ADK v2.3.0
    new_message = types.UserContent(
        parts=[types.Part(text=prompt)]
    )
    
    # Construct state delta for dynamic raw file path
    state_delta = None
    if raw_data_path:
        state_delta = {"raw_data_path": raw_data_path}
    
    text_chunks = []
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=new_message,
        state_delta=state_delta
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    text_chunks.append(part.text)
                    
    if not text_chunks:
        return "⚠️ No response text was returned by the Root Agent."
        
    return "".join(text_chunks)

def get_assistant_prompt(user_question, df_context, metrics_context, report_context):
    """
    Constructs a concise, business-focused prompt for the AI Assistant.
    Keeps the overall prompt length under ~3000 characters by omitting full CSV and business reports,
    focusing only on key metrics, reorder alerts, savings, top 10 reorder products, and top 5 suppliers.
    """
    total_products = df_context["Product"].nunique() if "Product" in df_context.columns else 0
    total_rows = len(df_context)
    
    reorder_alerts = 0
    if "Recommendation_Status" in df_context.columns:
        reorder_alerts = (df_context["Recommendation_Status"] == "REORDER").sum()
        
    original_cost = df_context["Reorder_Cost"].sum() if "Reorder_Cost" in df_context.columns else 0.0
    optimized_cost = df_context["Optimized_Reorder_Cost"].sum() if "Optimized_Reorder_Cost" in df_context.columns else 0.0
    cost_savings = max(0.0, original_cost - optimized_cost)
    
    # Get top 10 reorder products (Product, Recommendation_Status, Recommended_Order_Quantity)
    if "Recommendation_Status" in df_context.columns and "Product" in df_context.columns:
        reorder_df = df_context[df_context["Recommendation_Status"] == "REORDER"]
        if not reorder_df.empty:
            if "Recommended_Order_Quantity" in reorder_df.columns:
                top_reorder = reorder_df.sort_values(by="Recommended_Order_Quantity", ascending=False).head(10)
            else:
                top_reorder = reorder_df.head(10)
            
            cols_reorder = [c for c in ["Product", "Recommendation_Status", "Recommended_Order_Quantity"] if c in top_reorder.columns]
            top_reorder_str = top_reorder[cols_reorder].to_csv(index=False)
        else:
            top_reorder_str = "No products require reordering."
    else:
        top_reorder_str = "Reorder information not available."
        
    # Get top 5 optimized suppliers
    if "Optimized_Supplier" in df_context.columns:
        df_temp = df_context.copy()
        if "Reorder_Cost" in df_temp.columns and "Optimized_Reorder_Cost" in df_temp.columns:
            df_temp["Savings"] = df_temp["Reorder_Cost"] - df_temp["Optimized_Reorder_Cost"]
            supplier_summary = df_temp.groupby("Optimized_Supplier").agg(
                Times_Selected=("Optimized_Supplier", "count"),
                Total_Savings_USD=("Savings", "sum")
            ).reset_index()
            top_suppliers = supplier_summary.sort_values(by="Total_Savings_USD", ascending=False).head(5)
            top_suppliers["Total_Savings_USD"] = top_suppliers["Total_Savings_USD"].round(2)
        else:
            supplier_counts = df_context["Optimized_Supplier"].value_counts().head(5).reset_index()
            supplier_counts.columns = ["Optimized_Supplier", "Times_Selected"]
            top_suppliers = supplier_counts
            
        top_suppliers_str = top_suppliers.to_csv(index=False)
    else:
        top_suppliers_str = "Optimized supplier information not available."
        
    r2 = metrics_context.get("r2", 0.0)
    rmse = metrics_context.get("rmse", 0.0)
    mae = metrics_context.get("mae", 0.0)
    mse = metrics_context.get("mse", 0.0)

    prompt = f"""You are an AI Inventory Assistant. You help users understand their multi-agent inventory optimization results.
Below is the context containing results from the optimization pipeline.

--- 1. demand forecasting validation metrics ---
R² Accuracy: {r2*100:.2f}%
RMSE (Root Mean Squared Error): {rmse:.2f}
MAE (Mean Absolute Error): {mae:.2f}
MSE (Mean Squared Error): {mse:.2f}

--- 2. overall summary statistics ---
Total unique products: {total_products}
Total records processed: {total_rows}
Reorder Alerts: {reorder_alerts}
Total Sourcing Cost Savings: ${cost_savings:,.2f}

--- 3. top 10 reorder products ---
{top_reorder_str}

--- 4. top 5 optimized suppliers ---
{top_suppliers_str}

--- INSTRUCTIONS ---
- Answer the user's question clearly and concisely based ONLY on the provided context.
- Do NOT make assumptions, create or modify any data, or hallucinate results.
- If the information is not available in the context, state that you cannot answer based on current pipeline results.
- Keep answers professional, data-driven, and business-focused.

User's Question: {user_question}
Assistant Answer:"""
    return prompt


# Page configuration
st.set_page_config(
    page_title="AI Inventory Optimization Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for Power BI look and feel
st.markdown("""
    <style>
    .main {
        background-color: #F8F9FA;
    }
    .kpi-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        text-align: center;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        color: #1A365D;
        margin-top: 4px;
    }
    .kpi-label {
        font-size: 13px;
        font-weight: 500;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .section-header {
        font-size: 22px;
        font-weight: 600;
        color: #2D3748;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 8px;
        margin-top: 24px;
        margin-bottom: 16px;
    }
    .report-block {
        background-color: #FFFFFF;
        padding: 24px;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)


# Directories setup
RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

OPTIMIZED_PATH = os.path.join(PROCESSED_DIR, "optimized_inventory.csv")
METRICS_PATH = os.path.join(PROCESSED_DIR, "pipeline_metrics.json")
REPORT_PATH = os.path.join(PROCESSED_DIR, "inventory_business_report.md")


def run_complete_pipeline(raw_file_path: str):
    """
    Runs the multi-agent pipeline sequentially and saves forecasting evaluation metrics.
    """
    cleaned_path = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    inventory_path = os.path.join(PROCESSED_DIR, "inventory_recommendations.csv")

    # 1. Trigger Host Agent (Data Agent, Forecasting Agent, Inventory Agent)
    host_agent = HostAgent()
    host_results = host_agent.run_task({
        "raw_data_path": raw_file_path,
        "cleaned_data_path": cleaned_path,
        "inventory_output_path": inventory_path,
        "service_level_z": 1.65
    })

    if host_results.get("status") != "success":
        raise Exception(f"Pipeline failed at Host Agent step: {host_results.get('error')}")

    # Extract forecasting metrics
    forecasting_metrics = host_results.get("forecasting_agent_results", {}).get("metrics", {})
    if not forecasting_metrics:
        # Fallback if host agent was run in memory/cached mode
        from models.forecasting_model import ForecastingModel
        fm = ForecastingModel()
        forecasting_metrics = fm.train(cleaned_path)

    # 2. Trigger Optimization Agent
    opt_agent = OptimizationAgent()
    opt_results = opt_agent.run_task({
        "data_path": inventory_path,
        "output_path": OPTIMIZED_PATH,
        "service_level_z": 1.65
    })

    if opt_results.get("status") != "success":
        raise Exception(f"Pipeline failed at Optimization Agent step: {opt_results.get('error')}")

    # 3. Trigger Reporting Agent
    rep_agent = ReportingAgent()
    rep_results = rep_agent.run_task({
        "data_path": OPTIMIZED_PATH,
        "output_path": REPORT_PATH
    })

    if rep_results.get("status") != "success":
        raise Exception(f"Pipeline failed at Reporting Agent step: {rep_results.get('error')}")

    # Cache forecasting metrics to json for dashboard reload persistence
    pipeline_meta = {
        "r2": forecasting_metrics.get("r2", 0.0),
        "rmse": forecasting_metrics.get("rmse", 0.0),
        "mae": forecasting_metrics.get("mae", 0.0),
        "mse": forecasting_metrics.get("mse", 0.0),
        "last_updated": datetime.datetime.now().isoformat()
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(pipeline_meta, f, indent=2)

    return opt_results, rep_results, pipeline_meta


# Header Area
st.title("📊 AI Inventory Optimization Dashboard")
st.markdown("An enterprise multi-agent analytics tool implementing forecasting, replenishment modeling, supplier optimization, and Lagrangian warehouse capacity caps.")

# Sidebar Configuration and Upload
st.sidebar.header("Pipeline Controls")
uploaded_file = st.sidebar.file_uploader("Upload Inventory CSV File", type=["csv"])

pipeline_triggered = False
schema_valid = False

if uploaded_file is not None:
    raw_path = os.path.join(RAW_DIR, uploaded_file.name)
    
    try:
        # Read the uploaded CSV to validate columns
        uploaded_file.seek(0)
        df_uploaded = pd.read_csv(uploaded_file)
        
        # 1. Convert column names to expected format using normalization (lowercase, remove spaces & underscores)
        column_mapping = {
            "date": "Date",
            "product": "Product",
            "productname": "Product",
            "itemname": "Product",
            "category": "Category",
            "demand": "Demand",
            "price": "Price",
            "cost": "Cost",
            "warehouse": "Warehouse",
            "warehousename": "Warehouse",
            "supplier": "Supplier",
            "suppliername": "Supplier",
            "leadtime": "Lead_Time",
            "leadtimedays": "Lead_Time",
            "holdingcost": "Holding_Cost",
            "transportationcost": "Transportation_Cost",
            "storagecapacity": "Storage_Capacity",
            "storagecapacityunits": "Storage_Capacity",
            "currentstock": "Inventory_Level",
            "stock": "Inventory_Level",
            "inventorylevel": "Inventory_Level"
        }
        
        rename_map = {}
        for col in df_uploaded.columns:
            # Normalize column name: lowercase and remove spaces and underscores
            normalized_col = col.lower().replace(" ", "").replace("_", "")
            if normalized_col in column_mapping:
                rename_map[col] = column_mapping[normalized_col]
                
        df_uploaded = df_uploaded.rename(columns=rename_map)
        
        # 2. Check and handle derived columns if missing
        warnings = []
        if "Holding_Cost" not in df_uploaded.columns:
            if "Cost" in df_uploaded.columns:
                df_uploaded["Holding_Cost"] = (df_uploaded["Cost"].fillna(0.0) * 0.10).round(2)
                df_uploaded.loc[df_uploaded["Holding_Cost"] <= 0, "Holding_Cost"] = 5.0
            else:
                df_uploaded["Holding_Cost"] = 5.0
            warnings.append("⚠️ `Holding_Cost` column was missing and has been auto-created (calculated as 10% of Cost or default 5.0).")
            
        if "Transportation_Cost" not in df_uploaded.columns:
            if "Cost" in df_uploaded.columns:
                df_uploaded["Transportation_Cost"] = (df_uploaded["Cost"].fillna(0.0) * 0.50).round(2)
                df_uploaded.loc[df_uploaded["Transportation_Cost"] <= 0, "Transportation_Cost"] = 50.0
            else:
                df_uploaded["Transportation_Cost"] = 50.0
            warnings.append("⚠️ `Transportation_Cost` column was missing and has been auto-created (calculated as 50% of Cost or default 50.0).")
        
        # 3. Check for required columns
        required_cols = [
            "Date", "Product", "Category", "Demand", "Price", "Cost", "Warehouse",
            "Supplier", "Lead_Time", "Holding_Cost", "Transportation_Cost", "Storage_Capacity", "Inventory_Level"
        ]
        
        missing_cols = [col for col in required_cols if col not in df_uploaded.columns]
        
        if missing_cols:
            st.sidebar.error(
                "🚨 **Schema Validation Error**\n\n"
                "The uploaded CSV is missing the following required columns:\n"
                + "\n".join([f"- `{col}`" for col in missing_cols])
            )
            schema_valid = False
        else:
            st.sidebar.success("✅ CSV Schema validated successfully!")
            schema_valid = True
            
            # Show warnings when columns are auto-created
            if warnings:
                for warning in warnings:
                    st.sidebar.warning(warning)
            
            # Save the validated, formatted CSV file
            df_uploaded.to_csv(raw_path, index=False)
            
    except Exception as e:
        st.sidebar.error(f"🚨 **File Error**: Failed to process CSV: {str(e)}")
        schema_valid = False

    if st.sidebar.button("Run Multi-Agent Optimization Pipeline", type="primary", disabled=not schema_valid):
        if schema_valid:
            with st.status("Executing Multi-Agent Supply Chain Pipeline...", expanded=True) as status:
                try:
                    status.update(label="1. Invoking Host Agent (Data Cleaning, Forecast Training & Stock Targets)...")
                    st.write("🔧 Cleaning dataset, imputing missing values...")
                    st.write("📈 Training Random Forest Regressor & predicting demand...")
                    st.write("📐 Calculating Safety Stock, ROP, and EOQ limits...")
                    
                    # We execute
                    opt_res, rep_res, metrics_cache = run_complete_pipeline(raw_path)
                    
                    status.update(label="2. Invoking Optimization Agent (Supplier Selection & Warehouse Limits)...")
                    st.write("🤝 Sourcing cheapest suppliers based on transportation and lead time costs...")
                    st.write("🏢 Enforcing physical storage constraints via Lagrangian warehouse capacity caps...")
                    
                    status.update(label="3. Invoking Reporting Agent (KPI Aggregations & Markdown Report)...")
                    st.write("📊 Compiling dashboard metrics and writing executive report...")
                    
                    status.update(label="AI Pipeline executed successfully!", state="complete")
                    st.success("Optimization completed! Dashboard visuals have been refreshed.")
                    pipeline_triggered = True
                except Exception as e:
                    status.update(label="Pipeline execution failed!", state="error")
                    st.error(f"Error during agent coordination: {str(e)}")

# Load data and metrics for display
data_loaded = False
df = pd.DataFrame()
metrics = {"r2": 0.0, "rmse": 0.0, "mae": 0.0, "mse": 0.0}
report_content = ""

# Load existing results if present
if os.path.exists(OPTIMIZED_PATH):
    try:
        df = pd.read_csv(OPTIMIZED_PATH)
        data_loaded = True
        
        # Load cached metrics
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        else:
            # Bootstrap metrics if missing
            cleaned_path = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
            if os.path.exists(cleaned_path):
                from models.forecasting_model import ForecastingModel
                fm = ForecastingModel()
                metrics_calc = fm.train(cleaned_path)
                metrics = {
                    "r2": metrics_calc.get("r2", 0.0),
                    "rmse": metrics_calc.get("rmse", 0.0),
                    "mae": metrics_calc.get("mae", 0.0),
                    "mse": metrics_calc.get("mse", 0.0)
                }
                with open(METRICS_PATH, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)

        # Load business report markdown
        if os.path.exists(REPORT_PATH):
            with open(REPORT_PATH, "r", encoding="utf-8") as f:
                report_content = f.read()
    except Exception as e:
        st.warning(f"Could not load historical dashboard data: {str(e)}")

if not data_loaded:
    st.info("👋 Welcome! Please upload an inventory dataset in the sidebar and run the pipeline to start the dashboard.")
    st.stop()

# ----------------- DASHBOARD RENDER -----------------

# Calculate Dashboard level metrics
total_products = df["Product"].nunique()
total_rows = len(df)
reorder_alerts = (df["Recommendation_Status"] == "REORDER").sum()
total_opt_qty = df["Optimized_Order_Quantity"].sum()

# Sourcing Savings calculation
original_cost = df["Reorder_Cost"].sum()
optimized_cost = df["Optimized_Reorder_Cost"].sum()
cost_savings = max(0.0, original_cost - optimized_cost)

# KPI Cards row
kpi_cols = st.columns(7)

with kpi_cols[0]:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Products</div><div class="kpi-value">{total_products}</div></div>', unsafe_allow_html=True)
with kpi_cols[1]:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Processed Rows</div><div class="kpi-value">{total_rows}</div></div>', unsafe_allow_html=True)
with kpi_cols[2]:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Reorder Alerts</div><div class="kpi-value" style="color: #DD6B20;">{reorder_alerts}</div></div>', unsafe_allow_html=True)
with kpi_cols[3]:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Optimized Qty</div><div class="kpi-value">{int(total_opt_qty):,}</div></div>', unsafe_allow_html=True)
with kpi_cols[4]:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Cost Savings</div><div class="kpi-value" style="color: #38A169;">${cost_savings:,.2f}</div></div>', unsafe_allow_html=True)
with kpi_cols[5]:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">R² Accuracy</div><div class="kpi-value">{metrics.get("r2", 0.0)*100:.1f}%</div></div>', unsafe_allow_html=True)
with kpi_cols[6]:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Forecast RMSE</div><div class="kpi-value">{metrics.get("rmse", 0.0):.2f}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Demand Forecasting", 
    "📦 Inventory Recommendations", 
    "💰 Cost Sourcing & Capacity Optimization", 
    "📄 Executive Business Report",
    "💬 AI Inventory Assistant"
])

# ----------------- TAB 1: DEMAND FORECASTING -----------------
with tab1:
    st.markdown('<div class="section-header">Demand Forecast Evaluation & Timelines</div>', unsafe_allow_html=True)
    
    # Filter Controls
    prod_options = ["Network-Wide (All Products)"] + sorted(df["Product"].unique().tolist())
    selected_prod = st.selectbox("Select Product to Visualize", options=prod_options)

    # Prepare timeline plot data
    # Parse dates to sort chronologically
    df_timeline = df.copy()
    df_timeline["Parsed_Date"] = pd.to_datetime(df_timeline["Date"], format="%d-%m-%Y", errors="coerce")
    df_timeline = df_timeline.sort_values(by="Parsed_Date")

    if selected_prod == "Network-Wide (All Products)":
        # Group by date
        plot_df = df_timeline.groupby("Date").agg({
            "Parsed_Date": "first",
            "Demand": "sum",
            "predicted_demand": "sum"
        }).sort_values("Parsed_Date").reset_index()
        title_suffix = "Network-Wide (Total Demand)"
    else:
        plot_df = df_timeline[df_timeline["Product"] == selected_prod].reset_index()
        title_suffix = f"Product: {selected_prod}"

    # Demand line chart
    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Scatter(
        x=plot_df["Date"], y=plot_df["Demand"],
        mode="lines+markers", name="Actual Demand",
        line=dict(color="#3182CE", width=2.5),
        marker=dict(size=6)
    ))
    fig_timeline.add_trace(go.Scatter(
        x=plot_df["Date"], y=plot_df["predicted_demand"],
        mode="lines+markers", name="Predicted Demand",
        line=dict(color="#DD6B20", width=2.5, dash="dash"),
        marker=dict(size=6)
    ))
    fig_timeline.update_layout(
        title=f"Actual vs Predicted Demand Timeline - {title_suffix}",
        xaxis_title="Timeline Date",
        yaxis_title="Demand Quantity (Units)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#FFFFFF",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    fig_timeline.update_xaxes(showgrid=True, gridcolor="#E2E8F0")
    fig_timeline.update_yaxes(showgrid=True, gridcolor="#E2E8F0")
    
    st.plotly_chart(fig_timeline, use_container_width=True)

    # Split row for forecasting metrics card & forecast table
    cols_tab1 = st.columns([1, 2])
    with cols_tab1[0]:
        st.subheader("Model Validation Accuracy")
        st.markdown("""
        These validation metrics reflect the performance of the regression models evaluated on holdout splits during pipeline training.
        """)
        
        # Micro cards for other statistics
        sub_cols = st.columns(2)
        with sub_cols[0]:
            st.metric("MAE (Mean Absolute Error)", f"{metrics.get('mae', 0.0):.2f}")
        with sub_cols[1]:
            st.metric("MSE (Mean Squared Error)", f"{metrics.get('mse', 0.0):.1f}")
        
        sub_cols_2 = st.columns(2)
        with sub_cols_2[0]:
            st.metric("RMSE (Root Mean Sq. Error)", f"{metrics.get('rmse', 0.0):.2f}")
        with sub_cols_2[1]:
            st.metric("R² Score (Explained Var.)", f"{metrics.get('r2', 0.0):.3f}")

    with cols_tab1[1]:
        st.subheader("Product Forecast Stats Summary")
        forecast_stats = df.groupby("Product").agg(
            Average_Actual_Demand=("Demand", "mean"),
            Average_Forecasted_Demand=("predicted_demand", "mean"),
            Forecasted_Standard_Dev=("predicted_demand", "std"),
            Peak_Forecasted_Demand=("predicted_demand", "max")
        ).reset_index().round(2)
        st.dataframe(forecast_stats, hide_index=True, use_container_width=True)

# ----------------- TAB 2: INVENTORY RECOMMENDATIONS -----------------
with tab2:
    st.markdown('<div class="section-header">Inventory Recommendations & Stock Level Status</div>', unsafe_allow_html=True)
    
    cols_tab2_charts = st.columns(2)
    
    with cols_tab2_charts[0]:
        # Donut Chart for status distribution
        status_counts = df["Recommendation_Status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        
        colors_map = {"OK": "#3182CE", "REORDER": "#E53E3E", "OVERSTOCK": "#DD6B20"}
        fig_donut = px.pie(
            status_counts, values="Count", names="Status", hole=0.4,
            title="Distribution of Inventory Stock Statuses",
            color="Status",
            color_discrete_map=colors_map
        )
        fig_donut.update_traces(textinfo="percent+label")
        fig_donut.update_layout(showlegend=True, margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig_donut, use_container_width=True)

    with cols_tab2_charts[1]:
        # Bar Chart of Optimized Order Quantities by Product
        prod_order_df = df.groupby("Product")["Optimized_Order_Quantity"].sum().reset_index()
        prod_order_df = prod_order_df.sort_values("Optimized_Order_Quantity", ascending=False)
        
        fig_bar = px.bar(
            prod_order_df, x="Product", y="Optimized_Order_Quantity",
            title="Total Optimized Order Quantity by Product",
            labels={"Optimized_Order_Quantity": "Order Units"},
            color="Optimized_Order_Quantity",
            color_continuous_scale=px.colors.sequential.Blues
        )
        fig_bar.update_layout(plot_bgcolor="#FFFFFF", coloraxis_showscale=False)
        fig_bar.update_yaxes(showgrid=True, gridcolor="#E2E8F0")
        st.plotly_chart(fig_bar, use_container_width=True)

    # Detailed recommendations table
    st.subheader("Inventory Replenishment Configuration Log")
    rec_display_df = df[[
        "Product", "Warehouse", "Inventory_Level", "Safety_Stock", "Reorder_Point",
        "EOQ", "Recommendation_Status", "Recommended_Order_Quantity", "Optimized_Order_Quantity"
    ]].rename(columns={
        "Inventory_Level": "Current Stock",
        "Recommendation_Status": "Alert Status",
        "Recommended_Order_Quantity": "Original Order Qty (EOQ)",
        "Optimized_Order_Quantity": "Optimized Order Qty"
    })
    
    # Styled data frame with filter search
    st.dataframe(rec_display_df, use_container_width=True, hide_index=True)

# ----------------- TAB 3: COST SOURCING & CAPACITY OPTIMIZATION -----------------
with tab3:
    st.markdown('<div class="section-header">Financial Savings & Capacity Constraints Mitigations</div>', unsafe_allow_html=True)
    
    # Small Cards for savings metrics
    opt_kpi_cols = st.columns(4)
    with opt_kpi_cols[0]:
        st.metric("Total Original Order Cost", f"${original_cost:,.2f}")
    with opt_kpi_cols[1]:
        st.metric("Total Optimized Sourcing Cost", f"${optimized_cost:,.2f}")
    with opt_kpi_cols[2]:
        st.metric("Net Financial Savings", f"${cost_savings:,.2f}", delta=f"-{(original_cost - optimized_cost)/original_cost*100 if original_cost > 0 else 0.0:.2f}%", delta_color="inverse")
    with opt_kpi_cols[3]:
        # Enforced capacity corrections
        capacity_resolutions = ((df["Recommended_Order_Quantity"] != df["Optimized_Order_Quantity"]) & (df["Recommendation_Status"] == "REORDER")).sum()
        st.metric("Warehouse Capacity Enforcements", f"{capacity_resolutions} Items Scaled")

    cols_tab3_charts = st.columns(2)
    
    with cols_tab3_charts[0]:
        # Grouped bar chart comparing costs by category
        cat_costs = df.groupby("Category")[["Reorder_Cost", "Optimized_Reorder_Cost"]].sum().reset_index()
        fig_cat_costs = go.Figure()
        fig_cat_costs.add_trace(go.Bar(
            name="Original Order Cost", x=cat_costs["Category"], y=cat_costs["Reorder_Cost"],
            marker_color="#E53E3E"
        ))
        fig_cat_costs.add_trace(go.Bar(
            name="Optimized Sourcing Cost", x=cat_costs["Category"], y=cat_costs["Optimized_Reorder_Cost"],
            marker_color="#38A169"
        ))
        fig_cat_costs.update_layout(
            barmode="group",
            title="Order Cost Savings Breakdown by Product Category",
            yaxis_title="Total Cost ($)",
            plot_bgcolor="#FFFFFF",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_cat_costs.update_yaxes(showgrid=True, gridcolor="#E2E8F0")
        st.plotly_chart(fig_cat_costs, use_container_width=True)

    with cols_tab3_charts[1]:
        # Grouped bar chart comparing costs by warehouse
        wh_costs = df.groupby("Warehouse")[["Reorder_Cost", "Optimized_Reorder_Cost"]].sum().reset_index()
        fig_wh_costs = go.Figure()
        fig_wh_costs.add_trace(go.Bar(
            name="Original Order Cost", x=wh_costs["Warehouse"], y=wh_costs["Reorder_Cost"],
            marker_color="#E53E3E"
        ))
        fig_wh_costs.add_trace(go.Bar(
            name="Optimized Sourcing Cost", x=wh_costs["Warehouse"], y=wh_costs["Optimized_Reorder_Cost"],
            marker_color="#38A169"
        ))
        fig_wh_costs.update_layout(
            barmode="group",
            title="Order Cost Savings Breakdown by Warehouse Hub",
            yaxis_title="Total Cost ($)",
            plot_bgcolor="#FFFFFF",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_wh_costs.update_yaxes(showgrid=True, gridcolor="#E2E8F0")
        st.plotly_chart(fig_wh_costs, use_container_width=True)

    # Sourcing Supplier selection table
    st.subheader("Supplier Sourcing & Lead Time Audits")
    supplier_change_df = df[df["Recommendation_Status"] == "REORDER"][[
        "Product", "Warehouse", "Supplier", "Optimized_Supplier", "Cost", "Optimized_Cost",
        "Lead_Time", "Optimized_Lead_Time", "Reorder_Cost", "Optimized_Reorder_Cost"
    ]].copy()
    
    supplier_change_df["Direct_Savings"] = supplier_change_df["Reorder_Cost"] - supplier_change_df["Optimized_Reorder_Cost"]
    supplier_change_df = supplier_change_df.rename(columns={
        "Supplier": "Original Supplier",
        "Optimized_Supplier": "Optimized Supplier",
        "Cost": "Orig Unit Price ($)",
        "Optimized_Cost": "Opt Unit Price ($)",
        "Lead_Time": "Orig Lead Time",
        "Optimized_Lead_Time": "Opt Lead Time",
        "Reorder_Cost": "Original Cost",
        "Optimized_Reorder_Cost": "Optimized Cost",
        "Direct_Savings": "Sourcing Savings ($)"
    }).round(2)
    
    st.dataframe(supplier_change_df, use_container_width=True, hide_index=True)

# ----------------- TAB 4: EXECUTIVE BUSINESS REPORT -----------------
with tab4:
    st.markdown('<div class="section-header">Multi-Agent Business Report & Operational Risks</div>', unsafe_allow_html=True)
    
    if not report_content:
        st.info("Business report is not generated yet. Run the optimization pipeline to compile the report.")
    else:
        st.markdown("""
        Below is the business report generated by the **Reporting Agent** presenting summary metrics, capacity enforcements, and strategic sourcing recommendations.
        """)
        
        # Parse the markdown and organize it into clean visual expanders for Power BI layout
        lines = report_content.split("\n")
        sections = {}
        current_section = "General Overview"
        sections[current_section] = []
        
        for line in lines:
            if line.startswith("## "):
                current_section = line.replace("## ", "").strip()
                sections[current_section] = []
            elif line.startswith("# "):
                pass # Title is already shown at top
            else:
                sections[current_section].append(line)
        
        # Display the parsed sections as separate expanders
        for sec_name, sec_lines in sections.items():
            content_str = "\n".join(sec_lines).strip()
            if content_str:
                with st.expander(sec_name, expanded=(sec_name in ["1. Executive Summary", "4. Risk & Operational Alerts"])):
                    st.markdown(content_str)

# ----------------- TAB 5: AI INVENTORY ASSISTANT -----------------
with tab5:
    st.markdown('<div class="section-header">💬 AI Inventory Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        "Ask natural language questions about your forecasting accuracy, replenishment recommendations, "
        "supplier selection, cost savings, or physical warehouse capacity constraints."
    )
    
    # Reset chat if a new CSV was uploaded
    if uploaded_file is not None:
        if "last_uploaded_filename" not in st.session_state or st.session_state.last_uploaded_filename != uploaded_file.name:
            st.session_state.last_uploaded_filename = uploaded_file.name
            st.session_state.chat_messages = []
            st.session_state.adk_session_id = f"streamlit_session_{int(datetime.datetime.now().timestamp())}"
            
    # 1. Retrieve API key
    gemini_key = load_dotenv_key()
    
    if not gemini_key:
        st.warning(
            "🔑 **Gemini API Key missing!** Please add your API key as a `GEMINI_API_KEY` environment variable "
            "or create a `.env` file in the project root containing:\n"
            "```env\n"
            "GEMINI_API_KEY=\"your_api_key_here\"\n"
            "```"
        )
    else:
        # Initialize ADK session ID
        if "adk_session_id" not in st.session_state:
            st.session_state.adk_session_id = f"streamlit_session_{int(datetime.datetime.now().timestamp())}"

        # Initialize chat history
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
            
        # Display chat message history
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
        # Handle new user input
        if user_prompt := st.chat_input("Ask a question about the results (e.g., 'Which products need reorder?')..."):
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_prompt)
            st.session_state.chat_messages.append({"role": "user", "content": user_prompt})
            
            # Generate response from the ADK Root Agent
            with st.chat_message("assistant"):
                with st.spinner("Analyzing inventory results..."):
                    try:
                        session_id = st.session_state.adk_session_id
                        raw_data_path = None
                        if uploaded_file is not None:
                            raw_data_path = os.path.join(RAW_DIR, uploaded_file.name)
                        response_text = run_async_in_thread(call_adk_root_agent(user_prompt, session_id, raw_data_path))
                    except Exception as e:
                        err_msg = str(e)
                        if any(kw in err_msg for kw in ["429", "RESOURCE_EXHAUSTED", "Quota exceeded", "ResourceExhausted"]):
                            response_text = (
                                "⚠️ **Gemini API Quota Exceeded (429 RESOURCE_EXHAUSTED)**\n\n"
                                "Your Gemini API Key has exceeded its rate limit (RPM) or daily request quota (RPD).\n\n"
                                "**How to resolve this:**\n"
                                "1. **Wait a few seconds**: If it's a transient rate limit, waiting 10-30 seconds before asking again will resolve it.\n"
                                "2. **Daily Quota Limit**: If you are using the free tier, you may have reached the daily quota (e.g. 20 requests per day for `gemini-2.5-flash`).\n"
                                "3. **Check/Upgrade API Key**: Ensure you are using the correct API key in your `.env` file. You can upgrade to a pay-as-you-go key on [Google AI Studio](https://aistudio.google.com/) to increase rate limits and remove daily caps.\n\n"
                                f"**Error Details:**\n"
                                "```\n"
                                f"{err_msg.strip()}\n"
                                "```"
                            )
                        else:
                            response_text = f"❌ **ADK Runner Error**: Failed to run Root Agent. Details: {err_msg}"
                    st.markdown(response_text)
                    
            st.session_state.chat_messages.append({"role": "assistant", "content": response_text})