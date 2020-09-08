from flask import Flask, render_template, Response, request
from chatbot import get_robot_response

app = Flask(__name__)


@app.route('/')
def start():
    return render_template('start.html')


@app.route('/chat/<avatar>', methods=['GET', 'POST'])
def index(avatar):
    if avatar == 'maleImg':
        my_img = "https://i.ibb.co/TrKK2nr/download20200901194435.png"
        bot_img = "https://i.ibb.co/m0HCN2f/download20200901201411.png"
        my_color = '#fc808f'
        bot_color ='#007770'
    elif avatar == 'femaleImg':
        my_img = "https://i.ibb.co/m0HCN2f/download20200901201411.png"
        bot_img = "https://i.ibb.co/TrKK2nr/download20200901194435.png"
        my_color ='#007770'
        bot_color ='#fc808f'
    return render_template('index.html', avatar=my_img, bot_avatar=bot_img, my_color=my_color, bot_color=bot_color)


@app.route("/chat/get")
def get_bot_response():
    userText = request.args.get('msg')
    return get_robot_response(userText)


if __name__ == "__main__":
    app.run()
