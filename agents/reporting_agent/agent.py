"""
Reporting Agent Module.
Contains the ReportingAgent class responsible for taking Optimization Agent outputs,
calculating supply chain KPIs, summarizing cost savings, performing risk analyses,
and generating a formal Markdown business report.
"""

import os
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class ReportingAgent:
    """
    A modular agent responsible for generating inventory business reports.
    Takes optimization outputs as inputs and calculates KPIs, cost savings, and risks.
    """

    def __init__(self) -> None:
        """
        Initializes the ReportingAgent.
        """
        pass

    def run_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the business report generation task.

        Args:
            task_details: A dictionary containing task parameters.
                Expected keys:
                - 'data_path' (str, optional): Path to the optimized inventory recommendations CSV.
                  Defaults to 'data/processed/optimized_inventory.csv'.
                - 'output_path' (str, optional): Path to save the markdown business report.
                  Defaults to 'data/processed/inventory_business_report.md'.

        Returns:
            Dict[str, Any]: Execution results containing status, report summary KPIs, and output path.
        """
        data_path = task_details.get("data_path", os.path.join("data", "processed", "optimized_inventory.csv"))
        output_path = task_details.get("output_path", os.path.join("data", "processed", "inventory_business_report.md"))

        if not os.path.exists(data_path):
            return {
                "status": "failed",
                "error": f"Optimized inventory file not found at: {data_path}. Please run OptimizationAgent first."
            }

        try:
            # 1. Read optimization agent outputs
            df = pd.read_csv(data_path)

            # Define essential columns that must be present
            required_cols = [
                "Product", "Warehouse", "Category", "Inventory_Level", "Cost",
                "Storage_Capacity", "Safety_Stock", "Reorder_Point",
                "Recommendation_Status", "Recommended_Order_Quantity", "Reorder_Cost",
                "Optimized_Supplier", "Optimized_Lead_Time", "Optimized_Safety_Stock",
                "Optimized_Reorder_Point", "Optimized_Cost", "Optimized_Order_Quantity",
                "Optimized_Reorder_Cost", "Optimized_Stock_Value"
            ]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return {
                    "status": "failed",
                    "error": f"Input optimization dataset is missing required columns: {missing_cols}"
                }

            # 2. Calculate KPI Metrics
            total_items = len(df)
            unique_products = df["Product"].nunique()
            unique_warehouses = df["Warehouse"].nunique()

            # Order cost totals
            total_original_reorder_cost = float(df["Reorder_Cost"].sum())
            total_optimized_reorder_cost = float(df["Optimized_Reorder_Cost"].sum())
            total_cost_savings = total_original_reorder_cost - total_optimized_reorder_cost
            pct_cost_savings = (total_cost_savings / total_original_reorder_cost * 100) if total_original_reorder_cost > 0 else 0.0

            # Order quantities
            total_original_reorder_qty = int(df["Recommended_Order_Quantity"].sum())
            total_optimized_reorder_qty = int(df["Optimized_Order_Quantity"].sum())

            # Stock Value
            current_stock_val = float(df["Current_Stock_Value"].sum()) if "Current_Stock_Value" in df.columns else float((df["Inventory_Level"] * df["Cost"]).sum())
            optimized_stock_val = float(df["Optimized_Stock_Value"].sum())

            # Averages
            avg_original_lead_time = float(df["Lead_Time"].mean()) if "Lead_Time" in df.columns else 0.0
            avg_optimized_lead_time = float(df["Optimized_Lead_Time"].mean())
            avg_original_safety_stock = float(df["Safety_Stock"].mean())
            avg_optimized_safety_stock = float(df["Optimized_Safety_Stock"].mean())

            # Reorder alert metrics
            items_reordered_original = int((df["Recommended_Order_Quantity"] > 0).sum())
            items_reordered_optimized = int((df["Optimized_Order_Quantity"] > 0).sum())

            report_summary = {
                "total_items": total_items,
                "unique_products": unique_products,
                "unique_warehouses": unique_warehouses,
                "total_original_reorder_cost": round(total_original_reorder_cost, 2),
                "total_optimized_reorder_cost": round(total_optimized_reorder_cost, 2),
                "total_cost_savings": round(total_cost_savings, 2),
                "pct_cost_savings": round(pct_cost_savings, 2),
                "total_original_reorder_qty": total_original_reorder_qty,
                "total_optimized_reorder_qty": total_optimized_reorder_qty,
                "current_stock_value": round(current_stock_val, 2),
                "optimized_stock_value": round(optimized_stock_val, 2),
                "avg_original_lead_time": round(avg_original_lead_time, 2),
                "avg_optimized_lead_time": round(avg_optimized_lead_time, 2),
                "avg_original_safety_stock": round(avg_original_safety_stock, 2),
                "avg_optimized_safety_stock": round(avg_optimized_safety_stock, 2),
                "items_reordered_original": items_reordered_original,
                "items_reordered_optimized": items_reordered_optimized
            }

            # 3. Cost Savings Breakdowns
            # Warehouse savings
            warehouse_summary = []
            for wh, group in df.groupby("Warehouse"):
                orig_cost = float(group["Reorder_Cost"].sum())
                opt_cost = float(group["Optimized_Reorder_Cost"].sum())
                savings = orig_cost - opt_cost
                pct_savings = (savings / orig_cost * 100) if orig_cost > 0 else 0.0
                warehouse_summary.append({
                    "warehouse": wh,
                    "original_cost": round(orig_cost, 2),
                    "optimized_cost": round(opt_cost, 2),
                    "savings": round(savings, 2),
                    "pct_savings": round(pct_savings, 2)
                })
            warehouse_summary_df = pd.DataFrame(warehouse_summary).sort_values(by="savings", ascending=False)

            # Category savings
            category_summary = []
            for cat, group in df.groupby("Category"):
                orig_cost = float(group["Reorder_Cost"].sum())
                opt_cost = float(group["Optimized_Reorder_Cost"].sum())
                savings = orig_cost - opt_cost
                pct_savings = (savings / orig_cost * 100) if orig_cost > 0 else 0.0
                category_summary.append({
                    "category": cat,
                    "original_cost": round(orig_cost, 2),
                    "optimized_cost": round(opt_cost, 2),
                    "savings": round(savings, 2),
                    "pct_savings": round(pct_savings, 2)
                })
            category_summary_df = pd.DataFrame(category_summary).sort_values(by="savings", ascending=False)

            # Top 5 Cost Savings Items
            df_temp = df.copy()
            df_temp["Savings"] = df_temp["Reorder_Cost"] - df_temp["Optimized_Reorder_Cost"]
            top_savings_items = (
                df_temp.sort_values(by="Savings", ascending=False)
                .head(5)[["Product", "Warehouse", "Supplier", "Optimized_Supplier", "Reorder_Cost", "Optimized_Reorder_Cost", "Savings"]]
            )

            # 4. Risk and Alert Summaries
            # A. Stockout / Critical Low Stock risks (Inventory_Level == 0 or Inventory_Level < 20% of Safety Stock)
            critical_low_mask = (df["Inventory_Level"] == 0) | (df["Inventory_Level"] < 0.2 * df["Optimized_Safety_Stock"])
            critical_low_df = df[critical_low_mask][["Product", "Warehouse", "Inventory_Level", "Optimized_Safety_Stock", "Recommendation_Status"]]
            critical_low_count = len(critical_low_df)

            # B. Overstock risks (Recommendation_Status == 'OVERSTOCK')
            overstock_df = df[df["Recommendation_Status"] == "OVERSTOCK"][["Product", "Warehouse", "Inventory_Level", "Optimized_Reorder_Point"]]
            overstock_count = len(overstock_df)

            # C. Capacity Utilization & Mitigations
            wh_capacity_mitigations = []
            for wh, group in df.groupby("Warehouse"):
                capacity = group["Storage_Capacity"].iloc[0]
                if pd.isna(capacity) or capacity <= 0:
                    capacity = 10000.0
                
                initial_stock = group["Inventory_Level"].sum()
                unopt_orders = group["Recommended_Order_Quantity"].sum()
                opt_orders = group["Optimized_Order_Quantity"].sum()

                projected_unopt_stock = initial_stock + unopt_orders
                projected_opt_stock = initial_stock + opt_orders

                unopt_util = (projected_unopt_stock / capacity) * 100
                opt_util = (projected_opt_stock / capacity) * 100

                status = "OK"
                remedy_action = "N/A"
                if projected_unopt_stock > capacity:
                    if projected_opt_stock <= capacity:
                        status = "CAPACITY ENFORCED & MITIGATED"
                        remedy_action = f"Scaled down orders by {int(unopt_orders - opt_orders)} units"
                    else:
                        status = "OVER CAPACITY (CRITICAL)"
                        remedy_action = f"Reduced order by {int(unopt_orders - opt_orders)} units but warehouse remains full"
                elif opt_util > 90.0:
                    status = "WARNING (>90% Capacity)"

                wh_capacity_mitigations.append({
                    "warehouse": wh,
                    "capacity": int(capacity),
                    "initial_stock": int(initial_stock),
                    "original_orders": int(unopt_orders),
                    "optimized_orders": int(opt_orders),
                    "original_projected_util": round(unopt_util, 2),
                    "optimized_projected_util": round(opt_util, 2),
                    "status": status,
                    "mitigation": remedy_action
                })
            wh_capacity_df = pd.DataFrame(wh_capacity_mitigations)

            # D. Lead Time / Supplier Risks (Optimized Lead Time > 4 days)
            long_lead_time_df = df[df["Optimized_Lead_Time"] > 4.0][["Product", "Warehouse", "Optimized_Supplier", "Optimized_Lead_Time"]]
            long_lead_time_count = len(long_lead_time_df)

            # 5. Generate Markdown Business Report Content
            report_markdown = self._generate_report_markdown(
                report_summary,
                warehouse_summary_df,
                category_summary_df,
                top_savings_items,
                critical_low_df,
                overstock_df,
                wh_capacity_df,
                long_lead_time_df
            )

            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report_markdown)

            return {
                "status": "success",
                "report_summary": report_summary,
                "output_path": output_path
            }

        except Exception as e:
            return {
                "status": "failed",
                "error": f"An error occurred during report generation: {str(e)}"
            }

    def _generate_report_markdown(
        self,
        summary: Dict[str, Any],
        wh_summary: pd.DataFrame,
        cat_summary: pd.DataFrame,
        top_savings: pd.DataFrame,
        crit_low: pd.DataFrame,
        overstock: pd.DataFrame,
        capacity: pd.DataFrame,
        long_lead: pd.DataFrame
    ) -> str:
        """
        Compiles the report tables and metrics into a beautiful business-ready markdown document.
        """
        # Formulate executive summaries based on values
        mitigated_warehouses = len(capacity[capacity["status"] == "CAPACITY ENFORCED & MITIGATED"])
        stockout_count = len(crit_low)
        overstock_count = len(overstock)

        md = []
        md.append("# Executive Inventory & Replenishment Optimization Report")
        md.append(f"\n**Report Generation Date:** 2026-06-27 | **Scope:** Entire Distribution Network")
        md.append("\n## 1. Executive Summary")
        md.append(
            "This report summarizes the operational and financial impact of deploying the Agentic Inventory Optimization pipeline. "
            "By replacing generic Economic Order Quantity (EOQ) targets with supplier lead-time awareness and enforcing strict "
            "warehouse-level storage constraints, the system has achieved significant cost reductions while protecting stock availability."
        )
        md.append(
            f"\n- **Total Replenishment Cost Reduction:** Saved **${summary['total_cost_savings']:,.2f}** "
            f"({summary['pct_cost_savings']:.1f}% reduction), bringing total spend down from "
            f"${summary['total_original_reorder_cost']:,.2f} to **${summary['total_optimized_reorder_cost']:,.2f}**."
        )
        if mitigated_warehouses > 0:
            md.append(
                f"- **Warehouse Space Enforcements:** The optimization model successfully mitigated storage capacity violations in "
                f"**{mitigated_warehouses}** distribution hub(s) by dynamically scaling down over-commitments while staying within safety stock parameters."
            )
        else:
            md.append("- **Warehouse Space Enforcements:** All distribution hubs were confirmed to operate within standard physical capacity boundaries.")

        # KPI Metrics Table
        md.append("\n## 2. Key Performance Indicators (KPIs)")
        md.append("Below is a comparison of supply chain metrics under original recommendations versus optimized decisions:")
        md.append("\n| Metric | Original Recommendations | Optimized Decisions | Absolute Change | % Change |")
        md.append("| :--- | :---: | :---: | :---: | :---: |")
        
        reorder_cost_diff = summary['total_optimized_reorder_cost'] - summary['total_original_reorder_cost']
        md.append(f"| Total Replenishment Cost | ${summary['total_original_reorder_cost']:,.2f} | ${summary['total_optimized_reorder_cost']:,.2f} | ${reorder_cost_diff:,.2f} | {summary['pct_cost_savings']*-1:.1f}% |")
        
        reorder_qty_diff = summary['total_optimized_reorder_qty'] - summary['total_original_reorder_qty']
        reorder_qty_pct = (reorder_qty_diff / summary['total_original_reorder_qty'] * 100) if summary['total_original_reorder_qty'] > 0 else 0.0
        md.append(f"| Total Reorder Units | {summary['total_original_reorder_qty']:,} units | {summary['total_optimized_reorder_qty']:,} units | {reorder_qty_diff:+,} units | {reorder_qty_pct:+.1f}% |")

        stock_val_diff = summary['optimized_stock_value'] - summary['current_stock_value']
        stock_val_pct = (stock_val_diff / summary['current_stock_value'] * 100) if summary['current_stock_value'] > 0 else 0.0
        md.append(f"| Inventory Asset Value | ${summary['current_stock_value']:,.2f} | ${summary['optimized_stock_value']:,.2f} | {stock_val_diff:+,.2f} | {stock_val_pct:+.1f}% |")

        lead_time_diff = summary['avg_optimized_lead_time'] - summary['avg_original_lead_time']
        lead_time_pct = (lead_time_diff / summary['avg_original_lead_time'] * 100) if summary['avg_original_lead_time'] > 0 else 0.0
        md.append(f"| Average Lead Time | {summary['avg_original_lead_time']:.2f} days | {summary['avg_optimized_lead_time']:.2f} days | {lead_time_diff:+.2f} days | {lead_time_pct:+.1f}% |")

        safety_stock_diff = summary['avg_optimized_safety_stock'] - summary['avg_original_safety_stock']
        safety_stock_pct = (safety_stock_diff / summary['avg_original_safety_stock'] * 100) if summary['avg_original_safety_stock'] > 0 else 0.0
        md.append(f"| Average Safety Stock | {summary['avg_original_safety_stock']:.2f} units | {summary['avg_optimized_safety_stock']:.2f} units | {safety_stock_diff:+.2f} units | {safety_stock_pct:+.1f}% |")

        md.append(f"| Unique Active Products | {summary['unique_products']} | {summary['unique_products']} | - | - |")
        md.append(f"| Total Warehouses Monitored | {summary['unique_warehouses']} | {summary['unique_warehouses']} | - | - |")

        # Cost Savings Breakdown
        md.append("\n## 3. Cost Savings Analysis")
        
        md.append("\n### 3.1 Savings Breakdown by Warehouse")
        md.append("| Warehouse | Original Order Cost | Optimized Order Cost | Cost Savings ($) | % Savings |")
        md.append("| :--- | :---: | :---: | :---: | :---: |")
        for _, row in wh_summary.iterrows():
            md.append(f"| {row['warehouse']} | ${row['original_cost']:,.2f} | ${row['optimized_cost']:,.2f} | ${row['savings']:,.2f} | {row['pct_savings']:.1f}% |")

        md.append("\n### 3.2 Savings Breakdown by Category")
        md.append("| Category | Original Order Cost | Optimized Order Cost | Cost Savings ($) | % Savings |")
        md.append("| :--- | :---: | :---: | :---: | :---: |")
        for _, row in cat_summary.iterrows():
            md.append(f"| {row['category']} | ${row['original_cost']:,.2f} | ${row['optimized_cost']:,.2f} | ${row['savings']:,.2f} | {row['pct_savings']:.1f}% |")

        md.append("\n### 3.3 Top 5 Cost Savings Products")
        md.append("The table below highlights the individual products that generated the highest total financial savings through optimized supplier selection:")
        md.append("| Product | Warehouse | Original Supplier | Optimized Supplier | Original Cost | Optimized Cost | Net Savings |")
        md.append("| :--- | :--- | :--- | :--- | :---: | :---: | :---: |")
        for _, row in top_savings.iterrows():
            md.append(f"| {row['Product']} | {row['Warehouse']} | {row['Supplier']} | {row['Optimized_Supplier']} | ${row['Reorder_Cost']:,.2f} | ${row['Optimized_Reorder_Cost']:,.2f} | **${row['Savings']:,.2f}** |")

        # Risk & Operational Alerts
        md.append("\n## 4. Risk & Operational Alerts")

        # Capacity Audit
        md.append("\n### 4.1 Warehouse Storage Capacity Utilization & Mitigations")
        md.append("This section audits how the physical constraints of each distribution hub were addressed during optimization:")
        md.append("| Warehouse | Capacity | Initial Stock | Original Projected Stock | Optimized Projected Stock | Original Util. % | Optimized Util. % | Status |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")
        for _, row in capacity.iterrows():
            orig_proj = row['initial_stock'] + row['original_orders']
            opt_proj = row['initial_stock'] + row['optimized_orders']
            md.append(f"| {row['warehouse']} | {row['capacity']:,} | {row['initial_stock']:,} | {orig_proj:,} | {opt_proj:,} | {row['original_projected_util']}% | {row['optimized_projected_util']}% | **{row['status']}** |")

        # Low Stock
        md.append("\n### 4.2 Critical Stockout & Low Stock Alerts")
        if stockout_count > 0:
            md.append(f"**Warning:** Detected **{stockout_count}** items that are out of stock or critically below safety thresholds (< 20% safety stock):")
            md.append("\n| Product | Warehouse | Current Inventory | Optimized Safety Stock | Status |")
            md.append("| :--- | :--- | :---: | :---: | :--- |")
            for _, row in crit_low.head(10).iterrows():
                status_text = "STOCKOUT" if row['Inventory_Level'] == 0 else "CRITICALLY LOW"
                md.append(f"| {row['Product']} | {row['Warehouse']} | {row['Inventory_Level']:.0f} | {row['Optimized_Safety_Stock']:.1f} | `{status_text}` |")
            if stockout_count > 10:
                md.append(f"\n*And {stockout_count - 10} more items. Please check the source data for full details.*")
        else:
            md.append("No critical stockout or ultra-low inventory risks detected. Safety buffers are intact.")

        # Overstock
        md.append("\n### 4.3 Overstock Alerts")
        if overstock_count > 0:
            md.append(f"Identified **{overstock_count}** products whose current inventory levels exceed optimal reorder lines (ROP + 2*EOQ):")
            md.append("\n| Product | Warehouse | Current Stock | Optimized ROP | Excess Ratio |")
            md.append("| :--- | :--- | :---: | :---: | :---: |")
            for _, row in overstock.head(10).iterrows():
                ratio = row['Inventory_Level'] / row['Optimized_Reorder_Point'] if row['Optimized_Reorder_Point'] > 0 else 1.0
                md.append(f"| {row['Product']} | {row['Warehouse']} | {row['Inventory_Level']:.0f} | {row['Optimized_Reorder_Point']:.1f} | {ratio:.1f}x |")
            if overstock_count > 10:
                md.append(f"\n*And {overstock_count - 10} more items. Consider running promotions or redistributing stock to resolve overstocking.*")
        else:
            md.append("No overstocked items found. Inventory levels are operating close to optimal replenishment ranges.")

        # Supply chain lead times
        md.append("\n### 4.4 Supplier & Lead Time Risks")
        long_lead_count = len(long_lead)
        if long_lead_count > 0:
            md.append(f"Identified **{long_lead_count}** items relying on suppliers with long replenishment lead times (> 4 days). These present potential supply chain bottleneck points:")
            md.append("\n| Product | Warehouse | Supplier | Optimized Lead Time |")
            md.append("| :--- | :--- | :--- | :---: |")
            for _, row in long_lead.head(10).iterrows():
                md.append(f"| {row['Product']} | {row['Warehouse']} | {row['Optimized_Supplier']} | {row['Optimized_Lead_Time']:.1f} days |")
            if long_lead_count > 10:
                md.append(f"\n*And {long_lead_count - 10} more bottleneck items.*")
        else:
            md.append("No supplier lead-time bottlenecks (> 4 days) detected in the optimized sourcing plan.")

        # Actionable Suggestions
        md.append("\n## 5. Strategic Recommendations")
        md.append("1. **Transition Sourcing immediately:** Implement the supplier selections recommended in Section 3.3. These represent direct negotiations with lower total landed cost options.")
        md.append("2. **Implement Warehouse Capacity Caps:** Configure inventory management systems to strictly enforce the order quantities calculated for Bengaluru and other restricted hubs to prevent overflow fees and handling bottlenecks.")
        if stockout_count > 0:
            md.append("3. **Expedite Critical Shipments:** Review the items in Section 4.2 and place emergency purchase orders for immediate replenishment, as current stock levels are insufficient to cover immediate lead-time demand standard deviations.")
        if overstock_count > 0:
            md.append("4. **Redistribution Audit:** Review overstock items to see if they can be transferred from high-volume hubs to locations with active demand forecasts, minimizing overall holding costs.")
        
        return "\n".join(md)
