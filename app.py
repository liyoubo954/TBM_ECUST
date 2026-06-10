import logging
import os
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass

from app import create_app

app = create_app()
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
