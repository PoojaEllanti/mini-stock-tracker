from flask import Flask, render_template, request # type: ignore
import yfinance as yf # type: ignore
import plotly.graph_objs as go # type: ignore
import requests # type: ignore
from bs4 import BeautifulSoup # type: ignore

app = Flask(__name__)

def get_stock_data(symbol):
    stock = yf.Ticker(symbol)
    hist = stock.history(period="1mo")
    return hist

def create_chart(data, symbol):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode='lines', name='Close Price'))
    fig.update_layout(title=f"{symbol} Stock Price",
                      xaxis_title="Date",
                      yaxis_title="Price (USD)")
    return fig.to_html(full_html=False)

def predict_next_day(data):
    # Simple moving average prediction (last 5 days)
    if len(data) < 5:
        return "Not enough data to predict"
    prediction = data["Close"][-5:].mean()
    return f"${prediction:.2f} (Simple average)"

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
    return news  # Always return a list

@app.route("/", methods=['GET', 'POST'])
def index():
    chart = news = prediction = symbol = None

    if request.method == 'POST':
        symbol = request.form['symbol'].upper()
        df = get_stock_data(symbol)
        chart = create_chart(df, symbol)
        prediction = predict_next_day(df)
        news = get_news(symbol)

    return render_template("index.html", chart=chart, news=news or [], prediction=prediction, symbol=symbol)

if __name__ == "__main__":
    app.run(debug=True)
