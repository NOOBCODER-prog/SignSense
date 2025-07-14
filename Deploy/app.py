# I am using mediapipe as a hand landmark processing and prediction and landmark detector and a Random Forest classifier as sign classifier.
from wsgiref.simple_server import WSGIServer
from flask import (
    Flask,
    jsonify,
    render_template,
    url_for,
    redirect,
    flash,
    session,
    request,
    Response,
)
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    UserMixin,
    login_user,
    LoginManager,
    login_required,
    logout_user,
    current_user,
)
from flask_wtf import FlaskForm
import socketio
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError, Email, EqualTo
from flask_bcrypt import Bcrypt
from datetime import datetime
from itsdangerous.url_safe import URLSafeTimedSerializer as Serializer
from flask_mail import Message, Mail
import random
import re
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
from random import random
from english_words import english_words_lower_alpha_set

app = Flask(__name__)

CORS(app)

# -------------------Encrypt Password using Hash Func-------------------
bcrypt = Bcrypt(app)

# -------------------Database Model Setup-------------------
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SECRET_KEY"] = "someKey"
serializer = Serializer(app.config["SECRET_KEY"])
db = SQLAlchemy(app)
app.app_context().push()


login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# -------------------mail configuration-------------------
app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USERNAME"] = "test@gmail.com"
app.config["MAIL_PASSWORD"] = "password"
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USE_SSL"] = False
mail = Mail(app)
# --------------------------------------------------------


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# -------------------Database Model-------------------


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(30), nullable=False, unique=True)
    email = db.Column(db.String(30), nullable=False)
    password = db.Column(db.String(80), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)


# ----------------------------------------------------

# -------------------Welcome or Home Page-------------


@app.route("/", methods=["GET", "POST"])
def home():
    session.clear()
    return render_template("home.html")


# ----------------------------------------------------

# -------------------feed back Page-----------------------
@app.route("/feed", methods=["GET", "POST"])
@login_required
def feed():
    return render_template("feed.html")


# ----------------------------------------------------


# -------------------Login Page-------------------
class LoginForm(FlaskForm):
    username = StringField(
        label="username",
        validators=[InputRequired()],
        render_kw={"placeholder": "Username"},
    )
    email = StringField(
        label="email",
        validators=[InputRequired(), Email()],
        render_kw={"placeholder": "Email"},
    )
    password = PasswordField(
        label="password",
        validators=[InputRequired(), Length(min=8, max=20)],
        render_kw={"placeholder": "Password"},
    )
    submit = SubmitField("Login")


@app.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    # Check if the user has registered before showing the login form
    if "registered" in session and session["registered"]:
        session.pop("registered", None)
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if (
            user
            and bcrypt.check_password_hash(user.password, form.password.data)
            and User.query.filter_by(email=form.email.data).first()
        ):
            login_user(user)
            flash("Login successfully.", category="success")
            name = form.username.data
            session["name"] = name
            session["logged_in"] = True
            return redirect(url_for("dashboard"))
        else:
            flash(f"Login unsuccessful for {form.username.data}.", category="danger")
    return render_template("login.html", form=form)


# ----------------------------------------------------


# -------------------Dashboard or Logged Page-------------------
@app.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():
    if "logged_in" in session and session["logged_in"]:
        name = session.get("name")

        return render_template("dashboard.html", name=name)
    return redirect(url_for("login"))


# ----------------------------------------------------


# -------------------Logged Out Page-------------------


@app.route("/logout", methods=["GET", "POST"])
@login_required
def logout():
    session.clear()
    logout_user()
    flash("Account Logged out successfully.", category="success")
    return redirect(url_for("login"))


# ----------------------------------------------------

# -------------------Register Page-------------------


class RegisterForm(FlaskForm):
    username = StringField(
        label="username",
        validators=[InputRequired()],
        render_kw={"placeholder": "Username"},
    )
    email = StringField(
        label="email",
        validators=[InputRequired(), Email()],
        render_kw={"placeholder": "Email"},
    )
    password = PasswordField(
        label="password",
        validators=[InputRequired(), Length(min=8, max=20)],
        render_kw={"placeholder": "Password"},
    )
    confirm_password = PasswordField(
        label="confirm_password",
        validators=[InputRequired(), EqualTo("password")],
        render_kw={"placeholder": "Confirm Password"},
    )
    submit = SubmitField("Sign Up")

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(username=username.data).first()
        if existing_user_username:
            flash(
                "That Username already exists. Please choose a different one.", "danger"
            )
            raise ValidationError(
                "That username already exists. Please choose a different one."
            )


@app.route("/register", methods=["GET", "POST"])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(
            username=form.username.data, email=form.email.data, password=hashed_password
        )
        db.session.add(new_user)
        db.session.commit()
        # Set a session variable to indicate successful registration
        session["registered"] = True
        flash(
            f"Account Created for {form.username.data} successfully.",
            category="success",
        )
        return redirect(url_for("login"))

    return render_template("register.html", form=form)


# ----------------------------------------------------

# -------------------Update or reset Email Page-------------------


class ResetMailForm(FlaskForm):
    username = StringField(
        label="username",
        validators=[InputRequired()],
        render_kw={"placeholder": "Username"},
    )
    email = StringField(
        label="email",
        validators=[InputRequired(), Email()],
        render_kw={"placeholder": "Old Email"},
    )
    new_email = StringField(
        label="new_email",
        validators=[InputRequired(), Email()],
        render_kw={"placeholder": "New Email"},
    )
    password = PasswordField(
        label="password",
        validators=[InputRequired()],
        render_kw={"placeholder": "Password"},
    )
    submit = SubmitField("Login", validators=[InputRequired()])


@app.route("/reset_email", methods=["GET", "POST"])
@login_required
def reset_email():
    form = ResetMailForm()
    if "logged_in" in session and session["logged_in"]:
        if form.validate_on_submit():
            user = User.query.filter_by(username=form.username.data).first()
            if (
                user
                and bcrypt.check_password_hash(user.password, form.password.data)
                and User.query.filter_by(email=form.email.data).first()
            ):
                user.email = form.new_email.data  # Replace old email with new email
                db.session.commit()
                flash("Email reset successfully.", category="success")
                session.clear()
                return redirect(url_for("login"))
            else:
                flash("Invalid email, password, or combination.", category="danger")

        return render_template("reset_email.html", form=form)
    return redirect(url_for("login"))


# --------------------------------------------------------------

# -------------------Forgot Password With OTP-------------------


class ResetPasswordForm(FlaskForm):
    username = StringField(
        label="username",
        validators=[InputRequired()],
        render_kw={"placeholder": "Username"},
    )
    email = StringField(
        label="email",
        validators=[InputRequired(), Email()],
        render_kw={"placeholder": "Email"},
    )
    submit = SubmitField("Submit", validators=[InputRequired()])


class ForgotPasswordForm(FlaskForm):
    username = StringField(
        label="username",
        validators=[InputRequired()],
        render_kw={"placeholder": "Username"},
    )
    email = StringField(
        label="email",
        validators=[InputRequired(), Email()],
        render_kw={"placeholder": "Email"},
    )
    new_password = PasswordField(
        label="new_password",
        validators=[InputRequired(), Length(min=8, max=20)],
        render_kw={"placeholder": "New Password"},
    )
    confirm_password = PasswordField(
        label="confirm_password",
        validators=[InputRequired(), EqualTo("new_password")],
        render_kw={"placeholder": "Confirm Password"},
    )
    otp = StringField(
        label="otp",
        validators=[InputRequired(), Length(min=6, max=6)],
        render_kw={"placeholder": "Enter OTP"},
    )
    submit = SubmitField("Submit", validators=[InputRequired()])


@staticmethod
def send_mail(name, email, otp):
    msg = Message(
        "Reset Email OTP Password", sender="test@gmail.com", recipients=[email]
    )
    msg.body = "Hii " + name + "," + "\nYour email OTP is :" + str(otp)
    mail.send(msg)

    # Generate your OTP logic here


def generate_otp():
    return random.randint(100000, 999999)


@app.route("/reset_password_request", methods=["GET", "POST"])
def reset_password_request():
    otp = generate_otp()
    session["otp"] = otp
    form = ResetPasswordForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and User.query.filter_by(email=form.email.data).first():
            send_mail(form.username.data, form.email.data, otp)
            flash("Reset Request Sent. Check your mail.", "success")
            return redirect(url_for("forgot_password"))
        else:
            flash("Email and username combination is not exist.", "danger")
    return render_template("reset_password_request.html", form=form)


@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    form = ForgotPasswordForm()
    if form.validate_on_submit():
        otp = request.form["otp"]
        valid = otp == request.form["otp"]

        if valid:
            user = User.query.filter_by(username=form.username.data).first()
            if user and User.query.filter_by(email=form.email.data).first():
                user.password = bcrypt.generate_password_hash(
                    form.new_password.data
                ).decode("utf-8")
                db.session.commit()
                flash("Password Changed Successfully.", "success")
                return redirect(url_for("login"))
            else:
                flash("Email and username combination is not exist.", "danger")
        else:
            flash("OTP verification failed.", "danger")
    return render_template("forgot_password.html", form=form)


# ---------------------------------------------------------------

# ------------------------- Update Password ---------------------


class UpdatePasswordForm(FlaskForm):
    username = StringField(
        label="username",
        validators=[InputRequired()],
        render_kw={"placeholder": "Username"},
    )
    email = StringField(
        label="email",
        validators=[InputRequired(), Email()],
        render_kw={"placeholder": "Email"},
    )
    new_password = PasswordField(
        label="new_password",
        validators=[InputRequired(), Length(min=8, max=20)],
        render_kw={"placeholder": "New Password"},
    )
    confirm_password = PasswordField(
        label="confirm_password",
        validators=[InputRequired(), EqualTo("new_password")],
        render_kw={"placeholder": "Confirm Password"},
    )
    submit = SubmitField("Submit", validators=[InputRequired()])


@app.route("/update_password", methods=["GET", "POST"])
@login_required
def update_password():
    form = UpdatePasswordForm()
    if form.validate_on_submit() and "logged_in" in session and session["logged_in"]:

        user = User.query.filter_by(username=form.username.data).first()
        if user and User.query.filter_by(email=form.email.data).first():
            user.password = bcrypt.generate_password_hash(
                form.new_password.data
            ).decode("utf-8")
            db.session.commit()
            flash("Password Changed Successfully.", "success")
            session.clear()
            return redirect(url_for("login"))
        else:
            flash("Username and email combination is not exist.", "danger")
    return render_template("update_password.html", form=form)


# -----------------------------  end  ---------------------------


# --------------------------- Machine Learning ------------------
try:
    model_dict = pickle.load(open("./model.p", "rb"))
    model = model_dict["model"]
except Exception as e:
    print("Error loading the model:", e)
    model = None


@app.route("/generate_frames", methods=["POST"])
def generate_frames():
    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        max_num_hands=1, static_image_mode=True, min_detection_confidence=0.9
    )

    labels_dict = {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
        4: "E",
        5: "F",
        6: "G",
        7: "H",
        8: "I",
        10: "K",
        11: "L",
        12: "M",
        13: "N",
        14: "O",
        15: "P",
        16: "Q",
        17: "R",
        18: "S",
        19: "T",
        20: "U",
        21: "V",
        22: "W",
        23: "X",
        24: "Y",
    }
    test = ""

    last_update_time = time.time()
    update_interval = 1.5  # Delay in seconds

    while True:

        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            # for hand_landmarks in results.multi_hand_landmarks:
            #     mp_drawing.draw_landmarks(
            #         frame,  # image to draw
            #         hand_landmarks,  # model output
            #         mp_hands.HAND_CONNECTIONS,  # hand connections
            #         landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
            #         connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            data_aux += [0] * (84 - len(data_aux))

            try:
                prediction = model.predict([np.asarray(data_aux)])

                predicted_character = labels_dict[int(prediction[0])]

                # Check if enough time has elapsed to update the test string
                if time.time() - last_update_time >= update_interval:
                    test += predicted_character

                    if len(test) > 8:
                        test = ""

                    last_update_time = time.time()

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(
                    frame,
                    test,
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
            except Exception as e:
                pass

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

        # cv2.imshow('frame', frame)
        # cv2.waitKey(25)
        # if cv2.waitKey(25) == ord('q'):
        #     break


# --------------------- Learning Mode ------------------------------ #

words = [
    i
    for i in sorted(list(english_words_lower_alpha_set))
    if "z" not in i and len(i) > 3 and len(i) <= 10
]
start_time = time.time()
curr_time = 0
easy_word_user = ""
eraser = 0
easy_word = words[int(random() * len(words))].upper()
easy_word_index = 0
location = 0
letter_help = 0


def learning_mode(frame):
    global cap, easy_word_user, easy_word, easy_word_index, curr_time, location, letter_help

    def mediapipe_detection(image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
        results = model.process(image)  # Make prediction
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR CONVERSION RGB 2 BGR
        return image, results

    def get_landmark_dist_test(results, x, y):
        hand_array = []
        wrist_pos = results.multi_hand_landmarks[0].landmark[0]
        for result in results.multi_hand_landmarks[0].landmark:
            hand_array.append((result.x - wrist_pos.x) * (width / x))
            hand_array.append((result.y - wrist_pos.y) * (height / y))
        return hand_array[2:]

    # Main function
    # cap = cv2.VideoCapture(cam_max)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set mediapipe model
    mp_hands = mp.solutions.hands  # Hands model
    with mp_hands.Hands(
        min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1
    ) as hands:
        while cap.isOpened():

            # Read feed
            # ret, frame = cap.read()

            try:
                cv2.putText(
                    frame,
                    easy_word,
                    (int(width * 0.05), int(height * 0.95)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (255, 255, 255),
                    2,
                    cv2.LINE_4,
                )
                cv2.putText(
                    frame,
                    easy_word_user,
                    (int(width * 0.05), int(height * 0.95)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 255, 0),
                    2,
                    cv2.LINE_4,
                )
            except Exception as e:
                print(e)

            # Make detections
            image, results = mediapipe_detection(frame, hands)

            # Find bounding box of hand
            if results.multi_hand_landmarks:
                x = [None, None]
                y = [None, None]
                for result in results.multi_hand_landmarks[0].landmark:
                    if x[0] is None or result.x < x[0]:
                        x[0] = result.x
                    if x[1] is None or result.x > x[1]:
                        x[1] = result.x

                    if y[0] is None or result.y < y[0]:
                        y[0] = result.y
                    if y[1] is None or result.y > y[1]:
                        y[1] = result.y

                if (
                    curr_time < round((time.time() - start_time) / 3, 1)
                    and x[0] is not None
                ):
                    curr_time = round((time.time() - start_time) / 3, 1)
                    try:
                        test_image = get_landmark_dist_test(
                            results, x[1] - x[0], y[1] - y[0]
                        )
                        test_pred = np.argmax(clf.predict_proba(np.array([test_image])))
                        test_probs = clf.predict_proba(np.array([test_image]))[0]
                        print(
                            "Predicted:",
                            letters[test_pred],
                            ", pred prob:",
                            max(test_probs),
                            ", current index:",
                            easy_word_index,
                            ", current time:",
                            curr_time,
                        )
                        if max(test_probs) >= 0.8 or (
                            max(test_probs) >= 0.6
                            and letters[test_pred] in ["p", "r", "u", "v"]
                        ):
                            pred_letter = letters[test_pred].upper()
                            if (
                                easy_word_index < len(easy_word)
                                and pred_letter == easy_word[easy_word_index]
                                and (
                                    easy_word_index == 0
                                    or easy_word[easy_word_index]
                                    != easy_word[easy_word_index - 1]
                                )
                            ):
                                easy_word_user += pred_letter
                                easy_word_index += 1
                                location = results.multi_hand_landmarks[0].landmark[0].x
                            if (
                                easy_word_index < len(easy_word)
                                and pred_letter == easy_word[easy_word_index]
                                and easy_word_index > 0
                                and easy_word[easy_word_index]
                                == easy_word[easy_word_index - 1]
                                and abs(
                                    location
                                    - results.multi_hand_landmarks[0].landmark[0].x
                                )
                                > 0.1
                            ):
                                easy_word_user += pred_letter
                                easy_word_index += 1
                                location = results.multi_hand_landmarks[0].landmark[0].x

                        if easy_word_user == easy_word:
                            time.sleep(0.5)
                            easy_word = words[int(random() * len(words))].upper()
                            easy_word_index = 0
                            easy_word_user = ""

                    except Exception as e:
                        print(e)

            try:
                letter_help == 0
            except:
                frame[
                    5 : 5 + letter_help.shape[0],
                    width - 5 - letter_help.shape[1] : width - 5,
                ] = frame[
                    5 : 5 + letter_help.shape[0],
                    width - 5 - letter_help.shape[1] : width - 5,
                ]

            return frame

    return frame


def sign_frame():  # generate frame by frame from camera
    global easy, cap
    while True:
        success, frame = cap.read()
        if success:
            frame = medium_mode(frame)

        try:
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        except Exception as e:
            pass


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/learn_mode")
def sign_feed():
    return Response(sign_frame(), mimetype="multipart/x-mixed-replace; boundary=frame")


# -----------------------------  end  ---------------------------

if __name__ == "__main__":
    app.run(debug=True)
