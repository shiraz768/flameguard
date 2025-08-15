# auth.py
from flask import Blueprint, request, render_template, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from database import users_collection

auth_bp = Blueprint("auth", __name__)

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        if not username or not email or not password or not confirm_password:
            flash("All fields are required", "error")
            return redirect(url_for('auth.register'))

        if password != confirm_password:
            flash("Passwords do not match", "error")
            return redirect(url_for('auth.register'))

        if users_collection.find_one({"email": email}):
            flash("Email already registered", "error")
            return redirect(url_for('auth.register'))

        hashed_password = generate_password_hash(password)
        users_collection.insert_one({
            "username": username,
            "email": email,
            "password": hashed_password
        })

        flash("Registration successful! Please log in.", "success")
        return redirect(url_for('auth.login'))

    return render_template('register.html')


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get("email")
        password = request.form.get("password")

        if not email or not password:
            flash("Email and password are required", "error")
            return redirect(url_for('auth.login'))

        user = users_collection.find_one({"email": email})
        if not user or not check_password_hash(user["password"], password):
            flash("Invalid email or password", "error")
            return redirect(url_for('auth.login'))

       
        session['user_id'] = str(user['_id'])
        session['username'] = user['username']

        return redirect(url_for('home'))

    return render_template('login.html')


@auth_bp.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('auth.login'))
