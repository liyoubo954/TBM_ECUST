from flask import Flask


def create_app(config_class=None):
    app = Flask(__name__)
    if config_class is not None:
        app.config.from_object(config_class)

    from app.risk.routes import bp as risk_bp
    app.register_blueprint(risk_bp, url_prefix='/risk')

    return app
