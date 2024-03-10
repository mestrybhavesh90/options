pip install yfinance
pip install plotly
pip install pandas 


import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure Streamlit page layout
st.set_page_config(layout="wide")

# Function to download data based on selected interval and date range
def download_data(symbol, interval, start_date, end_date):
    ticker = yf.Ticker(symbol)
    df = ticker.history(interval=interval, start=start_date, end=end_date)
    return df

# Function to identify support level
def isSupport(df, i):
    support = df['Low'].iloc[i] < df['Low'].iloc[i-1] and df['Low'].iloc[i] < df['Low'].iloc[i+1] and df['Low'].iloc[i+1] < df['Low'].iloc[i+2] and df['Low'].iloc[i-1] < df['Low'].iloc[i-2]
    return support

# Function to identify resistance level
def isResistance(df, i):
    resistance = df['High'].iloc[i] > df['High'].iloc[i-1] and df['High'].iloc[i] > df['High'].iloc[i+1] and df['High'].iloc[i+1] > df['High'].iloc[i+2] and df['High'].iloc[i-1] > df['High'].iloc[i-2]
    return resistance

# Streamlit App

# Sidebar for inputs
st.sidebar.header('Input Parameters')
symbol = st.sidebar.text_input('Enter Symbol:', '^NSEI')
interval_options = ["1m", "5m", "15m", "30m", "1h"]  # Add more options if needed
selected_interval = st.sidebar.selectbox('Select Interval:', interval_options, index=1)
start_date = st.sidebar.date_input('Select Start Date:', pd.to_datetime("2024-03-03"))
end_date = st.sidebar.date_input('Select End Date:', pd.to_datetime("2024-03-10"))

# Download data based on selected interval and date range
df = download_data(symbol, selected_interval, start_date, end_date)

# List to store detected levels and their types (support or resistance)
levels = []

# Counters for touching support and resistance lines
support_touch_count = 0
resistance_touch_count = 0

# Identify support and resistance levels
for i in range(2, df.shape[0]-2):
    if isSupport(df, i):
        levels.append((i, df['Low'].iloc[i], 'support'))
        support_touch_count += 1
    elif isResistance(df, i):
        levels.append((i, df['High'].iloc[i], 'resistance'))
        resistance_touch_count += 1

# Plotly Figure with increased width and height
fig = go.Figure()

# Candlestick plot
candlestick = go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], showlegend=False)
fig.add_trace(candlestick)

# Support and Resistance levels
for level in levels:
    color = 'blue' if level[2] == 'support' else 'red'
    fig.add_trace(go.Scatter(x=[df.index[level[0]], df.index[-1]], y=[level[1], level[1]],
                             mode='lines', line=dict(color=color), name=f'{level[2].capitalize()} Level', showlegend=False))

# Update layout
fig.update_layout(title=f'Candlestick Chart with Support and Resistance for {symbol} ({selected_interval} interval)',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      xaxis_rangeslider_visible=True,  # Set x-axis rangeslider to False
                      width=1400,
                      height=800,
                      )

# Display the candlestick chart with support and resistance lines using Streamlit
st.plotly_chart(fig)

# Display the number of times candles touch support and resistance lines
st.sidebar.write(f"Number of times candles touch support line: {support_touch_count}")
st.sidebar.write(f"Number of times candles touch resistance line: {resistance_touch_count}")
