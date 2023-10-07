from flask import request
# from flask_login import current_user, login_user, logout_user
from weatherApi import app, db
from forms import LoginForm, RegistrationForm
from models import User
from config import Config
import requests

api_key = Config.API_KEY

@app.route('/')
def home():
    return 'Homepage!'

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password')
            return redirect(url_for('login'))
        login_user(user)
        return redirect(url_for('index'))
    return render_template('login.html', title='Sign In', form=form)

@app.route('/logout')
# def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
# def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Congratulations, you are now a registered user!')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)


@app.route('/current', methods=['POST'])
def get_current_weather():
    api_key = Config.API_KEY
    data = request.get_json()
    city = data.get('city', 'London')
    url = f'http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}&aqi=yes'

    response = requests.get(url)
    data = response.json()
    return data


@app.route('/forecast', methods=['POST'])
def get_forecast():
    api_key = Config.API_KEY
    data = request.get_json()
    city = data.get('city', 'London')
    days = data.get('days', '3')
    url = f'http://api.weatherapi.com/v1/forecast.json?key={api_key}&q={city}&days={days}&aqi=yes&alerts=yes'

    response = requests.get(url)
    data = response.json()
    return data


@app.route('/history', methods=['GET'])
def get_history():
    api_key = Config.API_KEY
    city = request.args.get('city', default='London', type=str)
    date = request.args.get('dt', default='2023-05-23', type=str)
    url = f'http://api.weatherapi.com/v1/history.json?key={api_key}&q={city}&dt={date}'

    response = requests.get(url)
    data = response.json()
    return data


@app.route('/future', methods=['GET'])
def get_future():
    api_key = Config.API_KEY
    city = request.args.get('city', default='London', type=str)
    date = request.args.get('dt', default='2023-05-23', type=str)
    url = f'http://api.weatherapi.com/v1/history.json?key={api_key}&q={city}&dt={date}'

    response = requests.get(url)
    data = response.json()
    return data


@app.route('/astronomy', methods=['GET'])
def get_astronomy():
    api_key = Config.API_KEY
    city = request.args.get('city', default='London', type=str)
    date = request.args.get('dt', default='2023-05-23', type=str)
    url = f'http://api.weatherapi.com/v1/astronomy.json?key={api_key}&q={city}&dt={date}'

    response = requests.get(url)
    data = response.json()
    return data
