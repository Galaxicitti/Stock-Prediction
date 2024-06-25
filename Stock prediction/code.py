import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Set page configuration
st.set_page_config(page_title="Stock Forecast App", layout="wide")

st.title('Stock Forecast App')

# Input fields for the stock symbol and start date
stock_symbol = st.text_input('Enter stock symbol')
start_date = "2015-01-01"

# Sliders for years, months, and days
n_years = st.slider('Years of prediction', 0, 4, 1)
n_months = st.slider('Months of prediction', 0, 11, 0)
n_days = st.slider('Days of prediction', 0, 30, 0)

# Calculate the total prediction period in days
total_days = n_years * 365 + n_months * 30 + n_days

# Cache data loading with the updated cache function
@st.cache_data
def load_data(symbol, start_date):
    data = yf.download(symbol, start=start_date, end=date.today().strftime("%Y-%m-%d"))
    data.reset_index(inplace=True)
    return data

# Check if inputs are provided
if stock_symbol and start_date:
    data_load_state = st.text('Loading data...')
    data = load_data(stock_symbol, start_date)
    data_load_state.text('Loading data... done!')

    st.subheader('Raw data')
    st.write(data.tail())

    # Plot raw data
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
        fig.update_layout(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()

    # Prepare the data for Prophet
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    # Fit the model
    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=total_days)
    forecast = m.predict(future)

    # Show and plot forecast
    st.subheader('Forecast data')
    st.write(forecast.tail())

    st.write(f'Forecast plot for {n_years} years, {n_months} months, and {n_days} days')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.write("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)
else:
    st.write("Please enter a stock symbol and select a start date.")
