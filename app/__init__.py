from flask import Flask
from config import Config

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    from app.risk import bp as risk_bp
    app.register_blueprint(risk_bp, url_prefix='/risk')
    
    return app
