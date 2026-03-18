import websocket
import json
import time
import hmac
import hashlib

API_KEY = "YOUR_API_KEY"
API_SECRET = "YOUR_SECRET_KEY"

def create_signature(timestamp):
    message = str(timestamp)
    signature = hmac.new(
        API_SECRET.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()
    return signature


def on_open(ws):
    print("Connected")

    timestamp = int(time.time() * 1000)
    signature = create_signature(timestamp)

    login_message = {
        "method": "login",
        "param": {
            "apiKey": API_KEY,
            "reqTime": timestamp,
            "signature": signature
        }
    }

    ws.send(json.dumps(login_message))


def on_message(ws, message):
    print("Received:", message)


def on_error(ws, error):
    print("Error:", error)


def on_close(ws, close_status_code, close_msg):
    print("Closed")


url = "wss://contract.mexc.com/ws"

ws = websocket.WebSocketApp(
    url,
    on_open=on_open,
    on_message=on_message,
    on_error=on_error,
    on_close=on_close
)

ws.run_forever()