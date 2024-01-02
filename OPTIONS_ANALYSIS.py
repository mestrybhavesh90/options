import streamlit as st
import pandas as pd
import requests
import time
from flask import Flask, render_template
import pandas as pd
from math import ceil
from datetime import datetime, timedelta
import numpy as np

st.set_page_config(layout="wide")

# Function to fetch option chain data
def fetch_option_chain_data():
    url = "https://www.nseindia.com/api/option-chain-indices?symbol=BANKNIFTY"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.5",
    }

    try:
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data_raw = response.json()
            records = data_raw.get("records", {})

            records = data_raw.get("records", {})

            latest_expiry_date = datetime.strptime(records["expiryDates"][0], "%d-%b-%Y")
            latest_expiry_date = latest_expiry_date.replace(hour=15, minute=29, second=29)

            latest_expiry_data = []
            option_data = []
            unique_strike_prices = set()

            for record in records["data"]:
                if record["expiryDate"] == records["expiryDates"][0]:
                    ce_data = record.get("CE", {})
                    pe_data = record.get("PE", {})

                    strikePrice = ce_data.get("strikePrice", 0)
                    
                    spotPrice = ce_data.get("underlyingValue", 0)

                    if strikePrice not in unique_strike_prices:
                        ce_LTP = ce_data.get("lastPrice", 0)
                        pe_LTP = pe_data.get("lastPrice", 0)

                        ce_OI = ce_data.get("openInterest", 0)
                        pe_OI = pe_data.get("openInterest", 0)

                        ce_chng_OI = ce_data.get("changeinOpenInterest", 0)
                        pe_chng_OI = pe_data.get("changeinOpenInterest", 0)

                        ce_IV = ce_data.get("impliedVolatility", 0)
                        pe_IV = pe_data.get("impliedVolatility", 0)

                        ce_volume = ce_data.get("totalTradedVolume", 0)
                        pe_volume = pe_data.get("totalTradedVolume", 0)

                        ce_totalBuyQty = ce_data.get("totalBuyQuantity", 0)
                        pe_totalBuyQty = pe_data.get("totalBuyQuantity", 0)
                        ce_totalSellQty = ce_data.get("totalSellQuantity", 0)
                        pe_totalSellQty = pe_data.get("totalSellQuantity", 0)

                        ce_delta, ce_vega, ce_theta, ce_gamma, ce_rho, pe_delta, pe_vega, pe_theta, pe_gamma, pe_rho = calculate_greeks(
                            strikePrice,
                            ce_LTP,
                            pe_LTP,
                            spotPrice,
                            ce_IV,
                            pe_IV,
                            latest_expiry_date
                        )

                        current_datetime = datetime.now()
                        latest_expiry_data = {
                            #'expiryDate': latest_expiry_date,
                            #'dateTime': current_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                            'spotPrice': spotPrice,
                            
                            'ce_LTP': ce_LTP,
                            'ce_IV': ce_IV,
                            'ce_volume': ce_volume,
                            'ce_chng_OI': ce_chng_OI,
                            'ce_OI': ce_OI,

                            #'ce_delta': ce_delta,
                            #'ce_theta': ce_theta,
                            #'ce_gamma': ce_gamma,
                            #'ce_vega': ce_vega,
                            #'ce_rho': ce_rho,
                            'ce_totalBuyQty': ce_totalBuyQty,
                            'ce_totalSellQty': ce_totalSellQty,

                            'strikePrice': strikePrice,

                            'pe_OI': pe_OI,
                            'pe_chng_OI': pe_chng_OI,
                            'pe_volume': pe_volume,
                            'pe_IV': pe_IV,
                            'pe_LTP': pe_LTP,

                            #'pe_delta': pe_delta,
                            #'pe_theta': pe_theta,
                            #'pe_gamma': pe_gamma,
                            #'pe_vega': pe_vega,
                            #'pe_rho': pe_rho,
                            'pe_totalBuyQty': pe_totalBuyQty,
                            'pe_totalSellQty': pe_totalSellQty,
                        }

                    option_data.append(latest_expiry_data)
                    unique_strike_prices.add(strikePrice)

            return option_data

        else:
            time.sleep(5)
            fetch_option_chain_data()

    except Exception as e:
        print("Error fetching data:", str(e))
        return []

def calculate_time_to_expiry(current_time, latest_expiry_date):
    
    # Define market open and close times
    market_open_time = datetime.strptime('09:15', '%H:%M').time()
    market_close_time = datetime.strptime('15:30', '%H:%M').time()

    # Initialize time_to_expiry_minutes
    time_to_expiry_minutes = 0
    
    # Iterate through each day between current time and latest expiry date
    current_date = current_time.date()  # Ignore the time component for comparison
    
    while current_date <= latest_expiry_date.date():  # Check only the date part
        # Check if it's a weekday (Monday to Friday)
        
        if current_date.weekday() < 5:
            # Calculate market open and close times for the current date
            
            market_open = datetime.combine(current_date, market_open_time)
            market_close = datetime.combine(current_date, market_close_time)

            # Adjust market open time for the current day
            market_open = max(market_open, current_time)

            # Adjust market close time for the expiry date
            if current_date == latest_expiry_date.date():
                # Special case for the last day
                market_close = min(market_close, latest_expiry_date)
            elif current_date == latest_expiry_date.date() - timedelta(days=1) and latest_expiry_date.weekday() < 5:
                # Special case for the day before expiry if it's a weekday
                market_close = min(market_close, latest_expiry_date)
            else:
                market_close = min(market_close, datetime.combine(current_date, datetime.max.time()))

            # Add the time difference for the current day in minutes
            time_to_expiry_minutes += max((market_close - market_open).total_seconds() / 60, 0)
        
        # Move to the next day
        current_date += timedelta(days=1)
        
    return ceil(time_to_expiry_minutes)


# Function to calculate greeks (if needed)
def calculate_greeks(strike, call_price, put_price, underlying_price, ce_implied_volatility, pe_implied_volatility, latest_expiry_date):
    # Calculate time to expiry in minutes
    current_time = datetime.now()
    time_to_expiry_minutes = calculate_time_to_expiry(current_time, latest_expiry_date)

    # Calculate Greeks with checks to avoid division by zero
    ce_delta = (call_price - put_price) / underlying_price if underlying_price != 0 else 0
    ce_gamma = ((call_price - put_price) / (underlying_price * ce_implied_volatility ** 2)) if (underlying_price != 0 and ce_implied_volatility != 0) else 0
    ce_vega = call_price * ce_implied_volatility if ce_implied_volatility != 0 else 0
    ce_theta = (-1 / 365) * call_price * ce_implied_volatility * (time_to_expiry_minutes ** 0.5) if ce_implied_volatility != 0 else 0

    # Put Greeks
    pe_delta = (call_price - put_price) / underlying_price if underlying_price != 0 else 0
    pe_gamma = -((call_price - put_price) / (underlying_price * pe_implied_volatility ** 2)) if (underlying_price != 0 and pe_implied_volatility != 0) else 0
    pe_vega = put_price * pe_implied_volatility if pe_implied_volatility != 0 else 0
    pe_theta = (1 / 365) * put_price * pe_implied_volatility * (time_to_expiry_minutes ** 0.5) if pe_implied_volatility != 0 else 0

    # Set rho to 0
    ce_rho = 0
    pe_rho = 0

    return (
        format(ce_delta, '.10f'),
        format(ce_vega, '.10f'),
        format(ce_theta, '.10f'),
        format(ce_gamma, '.10f'),
        ce_rho,
        format(pe_delta, '.10f'),
        format(pe_vega, '.10f'),
        format(pe_theta, '.10f'),
        format(pe_gamma, '.10f'),
        pe_rho
    )


# Fetch option chain data
option_data = fetch_option_chain_data()


# Initialize an empty DataFrame outside the loop to hold totals
totals_table = pd.DataFrame()
table_placeholder = st.empty()  # Create an empty placeholder

# Function to display totals table with comments
def display_totals_table():
    global totals_table  # Use the totals_table variable from the global scope

    while True:
        option_data = fetch_option_chain_data()  # Fetch option chain data

        if option_data:

            placeholder = st.empty()
            df = pd.DataFrame(option_data)
            totals = df.drop(columns=['strikePrice','spotPrice']).sum(numeric_only=True)
            totals['strikePrice'] = 0

            # Ensure 'strikePrice' remains an object type
            totals_frame = totals.to_frame().T.astype({'strikePrice': 'object'})
            totals_frame['Timestamp'] = pd.Timestamp.now()

            if totals_table.empty:
                totals_table = totals_frame  # Assign the first frame to the totals_table
            else:
                # Calculate comments for ce and pe
                prev_ce_LTP = totals_table.iloc[-1]['ce_LTP']
                prev_ce_chng_OI = totals_table.iloc[-1]['ce_chng_OI']
                prev_pe_LTP = totals_table.iloc[-1]['pe_LTP']
                prev_pe_chng_OI = totals_table.iloc[-1]['pe_chng_OI']

                totals_frame['ce_comments'] = np.where(
                    (totals_frame['ce_LTP'] > prev_ce_LTP), 'LTP ++',
                    np.where((totals_frame['ce_chng_OI'] > prev_ce_chng_OI), 'COI ++',
                             np.where((totals_frame['ce_LTP'] < prev_ce_LTP), 'LTP --',
                                      np.where((totals_frame['ce_chng_OI'] < prev_ce_chng_OI), 'COI --', '')
                                      )
                             )
                )
                
                totals_frame['pe_comments'] = np.where(
                    (totals_frame['pe_LTP'] > prev_pe_LTP), 'LTP ++',
                    np.where((totals_frame['pe_chng_OI'] > prev_pe_chng_OI), 'COI ++',
                             np.where((totals_frame['pe_LTP'] < prev_pe_LTP), 'LTP --',
                                      np.where((totals_frame['pe_chng_OI'] < prev_pe_chng_OI), 'COI --', '')
                                      )
                             )
                )

                totals_table = pd.concat([totals_table, totals_frame], ignore_index=True)  # Concatenate new totals

            # Calculate PCR for each entry
            totals_table['PCR'] = totals_table['pe_OI'] / totals_table['ce_OI']

            # Replace NaN or infinite values with 0
            totals_table['PCR'].replace([np.inf, -np.inf], np.nan, inplace=True)
            totals_table['PCR'].fillna(0, inplace=True)

            placeholder.table(totals_table)  # Display the totals table with PCR column

            time.sleep(180)  # Wait for 5 MINUTES before updating again
            placeholder.empty()

        else:
            st.write("Error fetching data. Please check your connection or try again later.")
            break


# Initialize an empty DataFrame outside the loop to hold totals
totals_table = pd.DataFrame()
table_placeholder = st.empty()  # Create an empty placeholder

def calculate_pcr_for_display_df(df):
    

    strike_wise_pcr = {}
    for strike in df['strikePrice'].unique():
        strike_df = df[df['strikePrice'] == strike]
        call_open_interest = strike_df['ce_OI'].sum()
        put_open_interest = strike_df['pe_OI'].sum()

        if call_open_interest != 0 and put_open_interest != 0:
            strike_wise_pcr[strike] = put_open_interest / call_open_interest
        else:
            strike_wise_pcr[strike] = 0  # Set PCR to 0 if either call or put OI is 0

    df['PCR'] = df['strikePrice'].map(strike_wise_pcr)
    return df




# Display data in Streamlit app
if option_data:

    placeholder = st.empty()

    df = pd.DataFrame(option_data)
    df = df.sort_values('strikePrice')

    atm_strike = df.iloc[(df['strikePrice'] - df['spotPrice']).abs().argsort()[:1]].iloc[0]['strikePrice']

    records_above_atm = df[df['strikePrice'] > atm_strike].head(6)
    records_below_atm = df[df['strikePrice'] < atm_strike].tail(6)
    atm_record = df[df['strikePrice'] == atm_strike]

    highlighted_atm = atm_record.style.apply(lambda x: ['background: red' if x.name == atm_record.index[0] else '' for i in x])

    display_df = pd.concat([records_above_atm, atm_record, records_below_atm]).drop(columns=['spotPrice'])

    # Calculate PCR for display_df
    display_df = calculate_pcr_for_display_df(display_df)

    # Sort display_df by 'strikePrice' column in ascending order
    display_df = display_df.sort_values('strikePrice')

    # Assuming atm_strike is the index label for the ATM strike price
    highlighted_atm = display_df.style.apply(lambda x: ['background: red' if x.name == atm_strike else '' for i in x], axis=1)

    
    st.write(highlighted_atm)

    totals = df.drop(columns=['spotPrice']).sum(numeric_only=True)
    totals['strikePrice'] = 0

    # Ensure 'strikePrice' remains an object type
    totals_frame = totals.to_frame().T.astype({'strikePrice': 'object'})
    
    # Adding timestamp column
    totals_frame['Timestamp'] = pd.Timestamp.now()

    display_df = pd.concat([display_df, totals_frame], ignore_index=True)

    #st.write(display_df.style.apply(lambda x: ['background: red' if x.name == atm_strike else '' for i in x]))
    # Display the updated totals_frame with expanded view
    

else:
    st.write("Error fetching data. Please check your connection or try again later.")

# Display totals table
display_totals_table()