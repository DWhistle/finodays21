from flask import Flask, redirect, url_for
from private.controller import scoring_api
from logging.config import fileConfig


class Config:
    JWT_SECRET_KEY = 'dev_key'
    JWT_ALGORITHM = 'HS256'
    JWT_LEEWAY = 10000
    JWT_TOKEN_LOCATION = 'cookies'
    JWT_ACCESS_COOKIE_NAME = 'JwtToken'
    JWT_COOKIE_CSRF_PROTECT = False
    SQLALCHEMY_DATABASE_URI = 'postgresql://alar:alar@db:5432/alar'
    SECRET_KEY = 'dev_key'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

application = Flask(__name__)
application.config.from_object(Config)

application.register_blueprint(scoring_api)