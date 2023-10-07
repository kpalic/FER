from flask import Blueprint

login_bp = Blueprint('login', __name__)
registration_bp = Blueprint('register', __name__)
location_bp = Blueprint('location', __name__)

from .login import *
from .registration import *
from .newLocation import *