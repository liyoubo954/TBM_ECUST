from flask import Flask


def create_app(config_class=None):
    app = Flask(__name__)
    if config_class is not None:
        app.config.from_object(config_class)

    from app.risk import routes as risk_routes
    from app.risk.misc_routes import register_misc_routes

    risk_bp = risk_routes.bp
    register_misc_routes(risk_bp, risk_routes)
    app.register_blueprint(risk_bp, url_prefix='/risk')

    return app
