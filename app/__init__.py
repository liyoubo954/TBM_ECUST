from flask import Flask
import os

def create_app(config_class=None):
    app = Flask(__name__)
    if config_class is not None:
        app.config.from_object(config_class)
    else:
        app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or 'sf9glGVa20pyM1NtdukZ'
    from app.risk import bp as risk_bp
    app.register_blueprint(risk_bp, url_prefix='/risk')
    
    return app
