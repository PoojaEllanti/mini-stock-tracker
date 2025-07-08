from flask import Flask, render_template, request # type: ignore
import yfinance as yf # type: ignore
import plotly.graph_objs as go # type: ignore
import requests # type: ignore
from bs4 import BeautifulSoup # type: ignore
import numpy as np
from sklearn.preprocessing import MinMaxScaler # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore

app = Flask(__name__)

def get_stock_data(symbol):
    stock = yf.Ticker(symbol)
    hist = stock.history(period="6mo")
    return hist

def create_chart(data, symbol):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode='lines', name='Close Price'))
    fig.update_layout(title=f"{symbol} Stock Price",
                      xaxis_title="Date",
                      yaxis_title="Price (USD)")
    return fig.to_html(full_html=False)

def prepare_lstm_data(data, sequence_length=60):
    close_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_prices)

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i - sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    return np.array(X), np.array(y), scaler

def train_lstm_model(X, y):
    X = X.reshape(X.shape[0], X.shape[1], 1)
    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=16, verbose=0)
    return model

def predict_with_lstm(data):
    if len(data) < 60:
        return None, None, "❌ Not enough data (needs 60+ days)."

    X, y, scaler = prepare_lstm_data(data)
    model = train_lstm_model(X, y)

    last_sequence = X[-1].reshape(1, X.shape[1], 1)
    predicted_scaled = model.predict(last_sequence)[0][0]
    predicted_price = scaler.inverse_transform([[predicted_scaled]])[0][0]

    actual_price = data['Close'].iloc[-1]
    return actual_price, predicted_price, None

def get_news(symbol):
    news = []
    try:
        url = f"https://finance.yahoo.com/quote/{symbol}?p={symbol}"
        page = requests.get(url, timeout=5)
        soup = BeautifulSoup(page.content, 'html.parser')
        items = soup.find_all('h3', class_='Mb(5px)')[:5]
        for item in items:
            a = item.find('a')
            if a and a.text:
                news.append({
                    'title': a.text.strip(),
                    'url': "https://finance.yahoo.com" + a['href']
                })
    except Exception as e:
        print(f"[ERROR] Failed to fetch news: {e}")
    return news

@app.route("/", methods=['GET', 'POST'])
def index():
    chart = news = prediction = symbol = None

    if request.method == 'POST':
        symbol = request.form['symbol'].upper()
        df = get_stock_data(symbol)

        if df.empty or 'Close' not in df.columns or df['Close'].dropna().empty:
            prediction = "❌ Unable to fetch stock data. Please try another symbol."
            chart = None
            news = []
        else:
            chart = create_chart(df, symbol)
            actual, predicted, error = predict_with_lstm(df)

            if error:
                last_price = df['Close'].dropna().iloc[-1]
                prediction = f"📉 Latest Price: ${last_price:.2f} (LSTM prediction not available)"
            else:
                prediction = f"📈 Actual: ${actual:.2f} | 🤖 Predicted: ${predicted:.2f}"

            news = get_news(symbol)

    return render_template("index.html", chart=chart, news=news or [], prediction=prediction, symbol=symbol)

if __name__ == "__main__":
    app.run(debug=True)
