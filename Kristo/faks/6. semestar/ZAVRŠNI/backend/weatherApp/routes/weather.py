from flask import Blueprint, request, jsonify
from config import Config
import requests

weather_bp = Blueprint('weather', __name__)

@weather_bp.route('/')
def home():
    return 'Homepage!'

@weather_bp.route('/current', methods=['POST'])
def get_current_weather():
    api_key = Config.API_KEY
    data = request.get_json()
    city = data.get('city', 'London')
    url = f'http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}&aqi=yes'

    response = requests.get(url)
    data = response.json()
    return data


@weather_bp.route('/forecast', methods=['POST'])
def get_forecast():
    api_key = Config.API_KEY
    data = request.get_json()
    city = data.get('city', 'London')
    days = data.get('days', '13')
    url = f'http://api.weatherapi.com/v1/forecast.json?key={api_key}&q={city}&days={days}&aqi=yes&alerts=yes'

    response = requests.get(url)
    data = response.json()
    return data


@weather_bp.route('/history', methods=['GET'])
def get_history():
    api_key = Config.API_KEY
    city = request.args.get('city', default='London', type=str)
    date = request.args.get('dt', default='2023-05-23', type=str)
    url = f'http://api.weatherapi.com/v1/history.json?key={api_key}&q={city}&dt={date}'

    response = requests.get(url)
    data = response.json()
    return data


@weather_bp.route('/future', methods=['GET'])
def get_future():
    api_key = Config.API_KEY
    city = request.args.get('city', default='London', type=str)
    date = request.args.get('dt', default='2023-05-23', type=str)
    url = f'http://api.weatherapi.com/v1/history.json?key={api_key}&q={city}&dt={date}'

    response = requests.get(url)
    data = response.json()
    return data


@weather_bp.route('/astronomy', methods=['GET'])
def get_astronomy():
    api_key = Config.API_KEY
    city = request.args.get('city', default='London', type=str)
    date = request.args.get('dt', default='2023-05-23', type=str)
    url = f'http://api.weatherapi.com/v1/astronomy.json?key={api_key}&q={city}&dt={date}'

    response = requests.get(url)
    data = response.json()
    return data

