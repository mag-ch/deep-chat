from flask import Flask, render_template, Response, request
from chatbot import get_robot_response
app = Flask(__name__)

@app.route('/')

def index():
    return render_template('index.html')
    # return 'hello'

@app.route("/get")
def get_bot_response():
 userText = request.args.get('msg')
 return get_robot_response(userText)

if __name__ == "__main__":
 app.run()
