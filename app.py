from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit
import pandas as pd
from nlp_pipeline import process_query_hybrid
import os


# 1. Flask app & SocketIO
app = Flask(__name__, static_folder="static")
socketio = SocketIO(app, cors_allowed_origins="*")


# 2. Load the sales data once (shared by every request)
DATA_PATH = os.path.join("data", "sales.csv")
sales_dataframe = pd.read_csv(DATA_PATH)
sales_dataframe["date"] = pd.to_datetime(sales_dataframe["date"])

 
# 3. Serve the pretty HTML/CSS/JS front-end
@app.route("/")
def serve_chat_ui():
    return send_from_directory("static", "index.html")

 
# 4. Real-time message handling
@socketio.on("user_message")
def reply_to_user(message_payload):
    user_text = message_payload.get("text", "").strip()
    if not user_text:
        return  # ignore empty lines

    # The heavy lifting lives in nlp_pipeline.py
    bot_html = process_query_hybrid(user_text, sales_dataframe)

    # Push the answer back to the browser instantly
    emit("bot_reply", {"html": bot_html})
 
# 5. Run locally (PyCharm → Run)
if __name__ == "__main__":
    socketio.run(app, host="127.0.0.1", port=5000, debug=True)
