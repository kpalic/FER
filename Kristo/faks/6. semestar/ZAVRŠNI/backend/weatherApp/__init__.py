from flask import Flask
from flask_cors import CORS
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect
from flask_jwt_extended import JWTManager
from config import Config
from routes import weather_bp
from forms import registration_bp
from forms import login_bp
from forms import location_bp
from models.user import User
from database import db

# login = LoginManager()
# login.login_view = 'login'

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    CORS(app, origins="http://localhost:3000", supports_credentials=True)
    jwt = JWTManager(app)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:P22hvetA@localhost/weatherapp'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY'] = 'P22hvetA'
    app.config['WTF_CSRF_ENABLED'] = False

    csrf = CSRFProtect(app)
    db.init_app(app)
    # login.init_app(app)

    with app.app_context():
        users = User.query.all()
        for user in users:
            print(user.username)

    app.register_blueprint(weather_bp)
    app.register_blueprint(registration_bp)
    app.register_blueprint(login_bp)
    app.register_blueprint(location_bp)

    return app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
