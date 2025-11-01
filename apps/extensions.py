# apps/extensions.py      apps/__init__.py에서 일부 이동
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_wtf import CSRFProtect


db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()
csrf=CSRFProtect()
login_manager.login_view = 'auth.login'
login_manager.login_message = 'login should be required'
