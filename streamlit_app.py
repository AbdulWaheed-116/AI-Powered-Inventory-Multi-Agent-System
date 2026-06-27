import os
import json
import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Import the existing agents
from agents.host_agent.agent import HostAgent
from agents.optimization_agent.agent import OptimizationAgent
from agents.reporting_agent.agent import ReportingAgent

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

if uploaded_file is not None:
    st.sidebar.success("File uploaded successfully!")
    raw_path = os.path.join(RAW_DIR, uploaded_file.name)
    
    # Save the file bytes
    with open(raw_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.sidebar.button("Run Multi-Agent Optimization Pipeline", type="primary"):
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
                st.write("🏢 Enforcing physical storage constraints via Lagrangian multiplier search...")
                
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
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Demand Forecasting", 
    "📦 Inventory Recommendations", 
    "💰 Cost Sourcing & Capacity Optimization", 
    "📄 Executive Business Report"
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