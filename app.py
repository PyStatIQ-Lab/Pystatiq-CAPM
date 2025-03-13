import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.pdfgen import canvas

# File path for the stocklist.xlsx
FILE_PATH = 'stocklist.xlsx'

# Function to fetch financial data and calculate growth
def get_financial_data(symbol):
    stock = yf.Ticker(symbol)
    income_statement = stock.financials.T  # Transposed for easier access
    quarterly_income_statement = stock.quarterly_financials.T  # Quarterly data
    balance_sheet = stock.balance_sheet.T  # Transposed for easier access
    cashflow = stock.cashflow.T  # Transposed for easier access

    try:
        # Safe extraction method using .get() to handle missing data
        revenue = income_statement.get('Total Revenue', [np.nan])[0]
        cogs = income_statement.get('Cost Of Revenue', [np.nan])[0]
        gross_profit = income_statement.get('Gross Profit', [np.nan])[0]
        operating_income = income_statement.get('Operating Income', [np.nan])[0]
        net_income = income_statement.get('Net Income', [np.nan])[0]
        
        # Extract balance sheet data
        cash_equivalents = balance_sheet.get('Cash And Cash Equivalents', [np.nan])[0]
        accounts_receivable = balance_sheet.get('Accounts Receivable', [np.nan])[0]
        inventory = balance_sheet.get('Inventory', [np.nan])[0]
        pp_and_e = balance_sheet.get('Property, Plant and Equipment', [np.nan])[0]
        goodwill = balance_sheet.get('Goodwill', [np.nan])[0]
        total_assets = balance_sheet.get('Total Assets', [np.nan])[0]
        total_liabilities = balance_sheet.get('Total Liabilities Net Minority Interest', [np.nan])[0]
        equity = balance_sheet.get('Total Equity Gross Minority Interest', [np.nan])[0]

        # Profitability Ratios
        gross_profit_margin = (gross_profit / revenue) * 100 if revenue else np.nan
        operating_profit_margin = (operating_income / revenue) * 100 if revenue else np.nan
        net_profit_margin = (net_income / revenue) * 100 if revenue else np.nan
        roe = (net_income / equity) * 100 if equity else np.nan
        roa = (net_income / total_assets) * 100 if total_assets else np.nan

        # Liquidity Ratios
        current_ratio = (balance_sheet.get('Current Assets', [np.nan])[0] /
                         balance_sheet.get('Current Liabilities', [np.nan])[0]) if balance_sheet.get('Current Liabilities', [np.nan])[0] else np.nan
        quick_ratio = ((balance_sheet.get('Current Assets', [np.nan])[0] - inventory) /
                       balance_sheet.get('Current Liabilities', [np.nan])[0]) if balance_sheet.get('Current Liabilities', [np.nan])[0] else np.nan

        # Leverage Ratios
        debt_to_equity = (total_liabilities / equity) if equity else np.nan
        debt_to_assets = (total_liabilities / total_assets) if total_assets else np.nan

        # Efficiency Ratios
        asset_turnover = (revenue / total_assets) if total_assets else np.nan
        inventory_turnover = (cogs / inventory) if inventory else np.nan
        receivables_turnover = (revenue / accounts_receivable) if accounts_receivable else np.nan

        # Valuation Ratios
        pe_ratio = stock.info.get('trailingPE', np.nan)
        pb_ratio = stock.info.get('priceToBook', np.nan)
        dividend_yield = stock.info.get('dividendYield', np.nan)

        # Cash Flow Ratios
        operating_cash_flow = cashflow.get('Operating Cash Flow', [np.nan])[0]
        free_cash_flow = operating_cash_flow - cashflow.get('Capital Expenditures', [np.nan])[0] if operating_cash_flow and cashflow.get('Capital Expenditures', [np.nan])[0] else np.nan

        # Calculate growth for each financial metric (annual growth)
        revenue_growth = ((revenue - income_statement.get('Total Revenue', [np.nan])[1]) / income_statement.get('Total Revenue', [np.nan])[1]) * 100 if len(income_statement) > 1 else np.nan
        cogs_growth = ((cogs - income_statement.get('Cost Of Revenue', [np.nan])[1]) / income_statement.get('Cost Of Revenue', [np.nan])[1]) * 100 if len(income_statement) > 1 else np.nan
        gross_profit_growth = ((gross_profit - income_statement.get('Gross Profit', [np.nan])[1]) / income_statement.get('Gross Profit', [np.nan])[1]) * 100 if len(income_statement) > 1 else np.nan
        operating_income_growth = ((operating_income - income_statement.get('Operating Income', [np.nan])[1]) / income_statement.get('Operating Income', [np.nan])[1]) * 100 if len(income_statement) > 1 else np.nan
        net_income_growth = ((net_income - income_statement.get('Net Income', [np.nan])[1]) / income_statement.get('Net Income', [np.nan])[1]) * 100 if len(income_statement) > 1 else np.nan

        # Price performance (30-day and 1-year)
        price_history = stock.history(period="1y")
        price_performance_30d = ((price_history['Close'][-1] - price_history['Close'][0]) / price_history['Close'][0]) * 100
        price_performance_1y = ((price_history['Close'][-1] - price_history['Close'][0]) / price_history['Close'][0]) * 100

        return {
            'Symbol': symbol,
            'Revenue Growth': revenue_growth,
            'Gross Profit Growth': gross_profit_growth,
            'Operating Income Growth': operating_income_growth,
            'Net Income Growth': net_income_growth,
            '30 Day Price Performance': price_performance_30d,
            '1 Year Price Performance': price_performance_1y
        }
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

# Function to generate PDF report
def generate_pdf(results_df, sheet_name, analyst_name):
    packet = BytesIO()
    c = canvas.Canvas(packet, pagesize=letter)
    
    # Title and metadata
    c.setFont("Helvetica-Bold", 18)
    c.drawString(30, 750, "Stock Financial Growth and Price Movement Analysis Dashboard")
    c.setFont("Helvetica", 12)
    c.drawString(30, 730, f"Sheet Name: {sheet_name}")
    c.drawString(30, 710, f"Research Analyst: {analyst_name}")
    c.drawString(30, 690, f"Date of Analysis: {datetime.now().strftime('%Y-%m-%d')}")
    c.drawString(30, 670, "Disclaimer: This analysis is for informational purposes only. It does not constitute investment advice.")
    
    # Insert Top 5 Stocks with Highest Revenue Growth
    c.setFont("Helvetica-Bold", 14)
    c.drawString(30, 650, "Top 5 Stocks with Highest Revenue Growth")
    c.setFont("Helvetica", 10)
    y_position = 630
    headers = ['Symbol', 'Revenue Growth', 'Gross Profit Growth', 'Operating Income Growth', 'Net Income Growth', '30 Day Price Performance', '1 Year Price Performance']
    for i, header in enumerate(headers):
        c.drawString(30 + i*80, y_position, header)
    y_position -= 20
    
    # Add data rows for Top 5 Revenue Growth
    for index, row in results_df.sort_values(by='Revenue Growth', ascending=False).head(5).iterrows():
        for i, col in enumerate(headers):
            c.drawString(30 + i*80, y_position, str(row[col]))
        y_position -= 20
        if y_position < 100:
            c.showPage()
            y_position = 750
    
    # Insert Top 5 Stocks with Lowest Price Performance
    c.setFont("Helvetica-Bold", 14)
    c.drawString(30, y_position - 20, "Top 5 Stocks with Lowest Price Performance in Last 1 Month")
    c.setFont("Helvetica", 10)
    y_position -= 40
    for i, header in enumerate(headers):
        c.drawString(30 + i*80, y_position, header)
    y_position -= 20
    
    # Add data rows for Lowest Price Performance
    for index, row in results_df.sort_values(by='30 Day Price Performance').head(5).iterrows():
        for i, col in enumerate(headers):
            c.drawString(30 + i*80, y_position, str(row[col]))
        y_position -= 20
        if y_position < 100:
            c.showPage()
            y_position = 750

    # Save PDF to buffer
    c.save()
    packet.seek(0)
    
    return packet

# Function to fetch stock symbols from the selected sheet
def get_stock_symbols(sheet_name):
    xl = pd.ExcelFile(FILE_PATH)
    df = xl.parse(sheet_name)
    return df['Symbol'].tolist()

# Streamlit Dashboard UI
def main():
    st.title("Stock Financial Growth and Price Movement Analysis Dashboard")
    
    # Select Sheet
    sheet_name = st.selectbox("Select Sheet", ['NIFTY50', 'NIFTYNEXT50', 'NIFTY100', 'NIFTY20', 'NIFTY500'])
    
    # Research Analyst Name
    analyst_name = st.text_input("Research Analyst Name")
    
    # Analyze Stocks Button
    if st.button("Analyze Stocks"):
        symbols = get_stock_symbols(sheet_name)
        results = []
        
        # Fetch financial data for each symbol
        for symbol in symbols:
            stock_data = get_financial_data(symbol)
            if stock_data:
                results.append(stock_data)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Display the results in a table
        st.subheader(f"Top 5 Stocks with Highest Revenue Growth")
        top_stocks_revenue_growth = results_df.sort_values(by='Revenue Growth', ascending=False).head(5)
        st.dataframe(top_stocks_revenue_growth[['Symbol', 'Revenue Growth', 'Gross Profit Growth', 'Operating Income Growth', 'Net Income Growth', '30 Day Price Performance', '1 Year Price Performance']])
        
        st.subheader(f"Top 5 Stocks with Lowest Price Performance in Last 1 Month")
        top_stocks_price_performance = results_df.sort_values(by='30 Day Price Performance').head(5)
        st.dataframe(top_stocks_price_performance[['Symbol', '30 Day Price Performance', 'Revenue Growth', 'Net Income Growth', 'Operating Income Growth']])
        
        # Generate PDF report and download link
        pdf_report = generate_pdf(results_df, sheet_name, analyst_name)
        st.download_button(
            label="Download PDF Report",
            data=pdf_report.getvalue(),
            file_name="financial_analysis_report.pdf",
            mime="application/pdf"
        )

if __name__ == "__main__":
    main()
