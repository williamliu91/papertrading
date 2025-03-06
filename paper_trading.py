import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import datetime

def app():
    # --- Constants ---
    PORTFOLIO_FILE = "portfolio.csv"
    DEFAULT_BALANCE = 100000
    TRANSACTION_FEE_PERCENTAGE = 0.002  # 0.2% transaction fee
    TRANSACTION_LOG_FILE = "transactions.csv"

    # --- Functions from stock.py ---
    def fetch_stock_data(sheet_url):
        """Fetches stock data from a Google Sheet."""
        try:
            response = requests.get(sheet_url)
            if response.status_code != 200:
                st.error("Failed to fetch the Google Sheet. Please check the URL and try again.")
                return None

            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table")

            if not table:
                st.warning("No table found on the page. Google Sheets may not be scrapable due to dynamic content loading.")
                return None

            rows = table.find_all("tr")

            data = []
            stock_name = "Unknown Stock"

            if rows:
                stock_name_cell = rows[1].find("td") if len(rows) > 1 else None
                if stock_name_cell:
                    stock_name = stock_name_cell.text.strip()

                headers = [header.text.strip() for header in rows[2].find_all("td")[:6]] if len(rows) > 2 else []
                if not headers or len(set(headers)) != len(headers):
                    headers = ["Date", "Open", "High", "Low", "Close", "Volume"]

                for row in rows[2:]:
                    cells = row.find_all("td")[:6]
                    if cells:
                        data.append([cell.text.strip() for cell in cells])

            if not data:
                st.warning("No data found in the table.")
                return None

            df = pd.DataFrame(data, columns=headers)
            df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
            df = df.dropna().sort_values(by="Date", ascending=True)

            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            return stock_name, df.dropna()

        except Exception as e:
            st.error(f"An error occurred: {e}")
            return None

    def calculate_mfi(df, period=14):
        """Calculates the Money Flow Index (MFI)."""
        df['Typical Price'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['Raw Money Flow'] = df['Typical Price'] * df['Volume']
        df['Positive Money Flow'] = df['Raw Money Flow'].where(df['Typical Price'] > df['Typical Price'].shift(1), 0)
        df['Negative Money Flow'] = df['Raw Money Flow'].where(df['Typical Price'] < df['Typical Price'].shift(1), 0)
        df['Positive Money Flow'] = df['Positive Money Flow'].rolling(window=period).sum()
        df['Negative Money Flow'] = df['Negative Money Flow'].rolling(window=period).sum()
        df['MFI'] = 100 - (100 / (1 + (df['Positive Money Flow'] / df['Negative Money Flow'])))
        return df

    def trading_strategy(data):
        """Generates buy/sell signals based on MFI."""
        data['Signal'] = 0
        for i in range(1, len(data)):
            if (data['MFI'].iloc[i - 1] < 20) and (data['MFI'].iloc[i] > 20):
                data.loc[data.index[i], 'Signal'] = 1
            elif (data['MFI'].iloc[i - 1] > 80) and (data['MFI'].iloc[i] < 80):
                data.loc[data.index[i], 'Signal'] = -1
        return data

    def plot_stock_chart(stock_name, df):
        """Plots the stock chart with MFI and buy/sell signals."""
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=(f'{stock_name} Stock Price with Buy/Sell Signals', 'Money Flow Index (MFI)'),
                            row_heights=[0.7, 0.3])

        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Stock Price', line=dict(color='blue')),
                      row=1, col=1)

        fig.add_trace(go.Scatter(x=df[df['Signal'] == 1]['Date'], y=df[df['Signal'] == 1]['Close'], mode='markers',
                                 marker=dict(color='green', size=10, symbol='triangle-up', line=dict(width=2, color='black')),
                                 name='Buy Signal'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df[df['Signal'] == -1]['Date'], y=df[df['Signal'] == -1]['Close'], mode='markers',
                                 marker=dict(color='red', size=10, symbol='triangle-down', line=dict(color='black')),
                                 name='Sell Signal'), row=1, col=1)

        fig.add_trace(go.Scatter(x=df['Date'], y=df['MFI'], mode='lines', name='MFI', line=dict(color='purple')),
                      row=2, col=1)

        fig.add_hline(y=80, line_dash="dash", line_color="red", row=2, col=1, annotation_text="Overbought (80)")
        fig.add_hline(y=20, line_dash="dash", line_color="green", row=2, col=1, annotation_text="Oversold (20)")

        fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig)

    # --- Functions from paper_trading.py (with modifications) ---
    def get_stock_data(stock_name, df):
        """Get latest stock data from the fetched DataFrame."""
        try:
            if df is None or df.empty:
                return None

            # Assuming the DataFrame is sorted by date (ascending), the last row contains the latest data
            latest_data = df.iloc[-1]

            info = {
                'symbol': stock_name.split(" ")[0],  # Extract the stock symbol from the name
                'name': stock_name,  # Use the full stock name from the sheet
                'current_price': latest_data['Close'],
                'volume': latest_data['Volume'],
                'open': latest_data['Open'],
                'high': latest_data['High'],
                'low': latest_data['Low']
            }
            return info
        except Exception as e:
            print(f"Error in get_stock_data: {e}")
            return None

    def load_portfolio_and_balance():
        """Loads portfolio and balance from CSV."""
        if os.path.exists(PORTFOLIO_FILE):
            try:
                data = pd.read_csv(PORTFOLIO_FILE)
                if "Balance" in data.columns:
                    balance = data["Balance"].iloc[0]
                    portfolio = data.drop(columns=["Balance"])
                    # Group by symbol and aggregate shares and purchase price
                    portfolio = portfolio.groupby('Symbol').agg({
                        'Shares': 'sum',
                        'Purchase Price': lambda x: (x * portfolio.loc[x.index, 'Shares']).sum() / portfolio.loc[x.index, 'Shares'].sum()
                    }).reset_index()
                    return portfolio, balance
            except Exception as e:
                st.error(f"Error loading portfolio: {e}. Starting with a fresh portfolio.")
                return pd.DataFrame(columns=["Symbol", "Shares", "Purchase Price"]), DEFAULT_BALANCE
        return pd.DataFrame(columns=["Symbol", "Shares", "Purchase Price"]), DEFAULT_BALANCE

    def save_portfolio_and_balance(portfolio, balance):
        """Saves portfolio and balance to CSV."""
        portfolio["Balance"] = [balance] + [None] * (len(portfolio) - 1)
        portfolio.to_csv(PORTFOLIO_FILE, index=False)

    def log_transaction(date, symbol, action, shares, price, fee, total):
        """Logs transactions to a CSV file."""
        new_row = pd.DataFrame([{
            'Date': date,
            'Symbol': symbol,
            'Action': action,
            'Shares': shares,
            'Price': price,
            'Fee': fee,
            'Total': total
        }])

        if os.path.exists(TRANSACTION_LOG_FILE):
            try:
                transaction_log = pd.read_csv(TRANSACTION_LOG_FILE)
                transaction_log = pd.concat([transaction_log, new_row], ignore_index=True)
            except Exception as e:
                st.error(f"Error loading transaction log: {e}")
                transaction_log = new_row #Start a new transaction log
        else:
             transaction_log = new_row #Start a new transaction log

        transaction_log.to_csv(TRANSACTION_LOG_FILE, index=False)

    def buy_stock(symbol, shares, current_price):
        """Buys stock and updates portfolio and balance."""
        cost = current_price * shares
        transaction_fee = cost * TRANSACTION_FEE_PERCENTAGE
        total_cost = cost + transaction_fee

        if st.session_state.balance >= total_cost:
            new_row = pd.DataFrame([{'Symbol': symbol, 'Shares': shares, 'Purchase Price': current_price}])
            st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], ignore_index=True)
            st.session_state.balance -= total_cost
            st.success(f"Bought {shares} shares of {symbol} for ${cost:.2f} + ${transaction_fee:.2f} fee")
            save_portfolio_and_balance(st.session_state.portfolio, st.session_state.balance)

            # Log the transaction
            log_transaction(datetime.datetime.now().isoformat(), symbol, "Buy", shares, current_price, transaction_fee, -total_cost)

        else:
            st.error("Insufficient balance.")

    def sell_stock(symbol, shares, current_price):
        """Sells stock and updates portfolio and balance."""
        # Find all rows with the given symbol
        symbol_rows = st.session_state.portfolio[st.session_state.portfolio['Symbol'] == symbol]

        if symbol_rows.empty:
            st.error(f"No shares of {symbol} in your portfolio.")
            return

        # Calculate total shares owned for the given symbol
        total_shares_owned = symbol_rows['Shares'].sum()

        if total_shares_owned < shares:
            st.error(f"You only own {total_shares_owned} shares of {symbol}.")
            return

        # Calculate the proceeds from the sale
        proceeds = current_price * shares
        transaction_fee = proceeds * TRANSACTION_FEE_PERCENTAGE
        net_proceeds = proceeds - transaction_fee

        st.session_state.balance += net_proceeds
        st.success(f"Sold {shares} shares of {symbol} for ${proceeds:.2f} - ${transaction_fee:.2f} fee")

        # Remove shares from the portfolio
        shares_to_remove = shares
        rows_to_drop = []
        for index, row in symbol_rows.iterrows():
            if shares_to_remove >= row['Shares']:
                # Remove the entire row
                shares_to_remove -= row['Shares']
                rows_to_drop.append(index)
            else:
                # Update the row with the remaining shares
                st.session_state.portfolio.loc[index, 'Shares'] -= shares_to_remove
                shares_to_remove = 0
                break

        # Drop the rows that need to be removed
        st.session_state.portfolio = st.session_state.portfolio.drop(rows_to_drop)

        # If shares_to_remove is still > 0, it means there was a problem
        if shares_to_remove > 0:
            st.error("An unexpected error occurred while selling shares.")
            return

        # Save the updated portfolio and balance
        save_portfolio_and_balance(st.session_state.portfolio, st.session_state.balance)

        # Log the transaction
        log_transaction(datetime.datetime.now().isoformat(), symbol, "Sell", shares, current_price, transaction_fee, net_proceeds)

    # --- Main Streamlit App ---
    # Title and Header
    st.title("Stock Analysis and Paper Trading")

    # --- Load Portfolio and Balance ---
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = pd.DataFrame(columns=["Symbol", "Shares", "Purchase Price"])
    if 'balance' not in st.session_state:
        st.session_state.balance = DEFAULT_BALANCE  # Default balance if no file exists

    st.session_state.portfolio, st.session_state.balance = load_portfolio_and_balance()

    # --- Sidebar for Google Sheet URL ---
    st.sidebar.header("Google Sheet Data")
    sheet_url = st.sidebar.text_input("Enter Google Sheet URL:")
    fetch_data_button = st.sidebar.button("Fetch Data from Google Sheet")

    # --- Fetch Data from Google Sheet ---
    if fetch_data_button:
        if sheet_url:
            result = fetch_stock_data(sheet_url)
            if result:
                stock_name, df = result
                df = calculate_mfi(df)
                df = trading_strategy(df)
                st.session_state.stock_data = (stock_name, df)
            else:
                st.error("Failed to fetch data from the Google Sheet.")
        else:
            st.warning("Please enter a Google Sheet URL.")

    # --- Display Stock Data and Chart ---
    if st.session_state.get('stock_data'):
        stock_name, df = st.session_state.stock_data

        st.subheader(f"Stock Data: {stock_name}")

        # Display Latest 5 Days of Stock Data
        st.write("Latest 5 Days of Stock Data:")
        filtered_df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        filtered_df[numeric_cols] = filtered_df[numeric_cols].round(2)

        st.write(filtered_df.tail(5).style.format({
            'Open': '{:.2f}',
            'High': '{:.2f}',
            'Low': '{:.2f}',
            'Close': '{:.2f}'
        }))

        plot_stock_chart(stock_name, df)

        # --- Paper Trading Section using Google Sheet Stock Symbol---
        st.header("Paper Trading")
        st.markdown("""
            Monitor real-time market prices and engage in paper trading.
            You currently have a virtual balance of **${:,.2f}**.
        """.format(st.session_state.balance))

        # Stock Symbol Input for Paper Trading from Google sheet
        # symbol = stock_name.split(" ")[0] # Take the first word

        # if symbol:
        #     stock_data = get_stock_data(symbol) #OLD LINE
        #     if stock_data:

        stock_data = get_stock_data(stock_name, df)  # NEW LINE
        if stock_data:
            st.subheader(f"Trading {stock_data['name']} ({stock_data['symbol']})")
            st.write(f"Current Price: ${stock_data['current_price']:.2f}")

            # Buy/Sell Options
            col1, col2 = st.columns(2)
            with col1:
                shares_to_buy = st.number_input("Shares to Buy", min_value=1, step=1)
                if st.button("Buy"):
                    buy_stock(stock_data['symbol'], shares_to_buy, stock_data['current_price']) #used the symbol from stock_data
            with col2:
                shares_to_sell = st.number_input("Shares to Sell", min_value=1, step=1)
                if st.button("Sell"):
                    sell_stock(stock_data['symbol'], shares_to_sell, stock_data['current_price']) #used the symbol from stock_data
        else:
            st.error(f"Could not fetch data for {stock_name}.  Please verify the stock symbol.")

    else:
        st.info("Enter a Google Sheet URL and Fetch Data to display stock information.")

    # --- Display Portfolio ---
    st.subheader("Current Portfolio")
    if not st.session_state.portfolio.empty:
        st.dataframe(st.session_state.portfolio)
    else:
        st.info("Your portfolio is currently empty.")

    st.write(f"**Current Balance: ${st.session_state.balance:,.2f}**")

    # --- Display Transaction Log ---
    st.subheader("Transaction Log")
    if os.path.exists(TRANSACTION_LOG_FILE):
        try:
            transaction_log = pd.read_csv(TRANSACTION_LOG_FILE)
            st.dataframe(transaction_log)
        except Exception as e:
            st.error(f"Error loading transaction log: {e}")
    else:
        st.info("No transactions have been made yet.")

if __name__ == "__main__":
    app()