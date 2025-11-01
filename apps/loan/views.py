# apps/loan/views.py
from flask import render_template, current_app, request
#from flask_login import login_required, current_user
from apps.loan import loan

@loan.route("/")
def index():
    pass
@loan.route("/services")
def services():
    pass