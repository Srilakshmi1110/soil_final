import os
import sys
import json
import time
import sqlite3
import threading
import hashlib
import secrets
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import urlopen
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import joblib
import serial
import rasterio
from rasterio.warp import transform as warp_transform

from flask import Flask, jsonify, request, send_from_directory, session
from flask_cors import CORS

# ============================================================
# PROJECT PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = BASE_DIR / "backend"
MODEL_DIR = BACKEND_DIR / "models"
DATA_DIR = BACKEND_DIR / "data"
FRONTEND_DIR = BASE_DIR / "frontend"
DB_FILE = DATA_DIR / "predictions.db"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config

SERIAL_PORT = getattr(config, "SERIAL_PORT", "COM10")
SERIAL_BAUD = int(getattr(config, "SERIAL_BAUD", 9600))

app = Flask(
    __name__,
    static_folder=str(FRONTEND_DIR),
    static_url_path=""
)

app.secret_key = getattr(
    config,
    "SECRET_KEY", 
    "soilai-local-development-secret"
)

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax"
)

CORS(app, supports_credentials=True)

# ============================================================
# THREAD LOCKS AND RUNTIME STATE
# ============================================================

DB_LOCK = threading.RLock()
STATE_LOCK = threading.RLock()

arduino = None
arduino_lock = threading.Lock()

current = {
    "connected": False,
    "reading": None,
    "prediction": None,
    "explanation": None,
    "timestamp": None,
    "location_source": None,
    "error": "Waiting for Arduino data."
}

# ============================================================
# MODEL LOADING
# ============================================================

lgb_model = None
cat_model = None
background_data = None
class_names = []
FEATURES = []

shap_explainer = None
lime_explainer = None

model_error = None


def load_first_existing(directory, filenames):
    for filename in filenames:
        path = directory / filename
        if path.exists():
            return joblib.load(path), path
    raise FileNotFoundError(
        "Could not find any of these model files: "
        + ", ".join(filenames)
    )


def load_models():
    global lgb_model
    global cat_model
    global background_data
    global class_names
    global FEATURES
    global shap_explainer
    global lime_explainer
    global model_error

    try:
        feature_data, feature_path = load_first_existing(
            MODEL_DIR,
            [
                "feature_names_temperature.joblib",
                "feature_names.joblib"
            ]
        )

        FEATURES = list(feature_data)

        if not FEATURES:
            raise ValueError("The feature metadata file is empty.")

        if len(FEATURES) not in (6, 7):
            raise ValueError(
                f"Unsupported model feature count: {len(FEATURES)}"
            )

        if len(FEATURES) == 7:
            lgb_names = [
                "lgb_model_temperature.joblib",
                "lgb_model.joblib"
            ]
            cat_names = [
                "cat_model_temperature.joblib",
                "cat_model.joblib"
            ]
            background_names = [
                "background_data_temperature.joblib",
                "background_data.joblib"
            ]
            class_names_files = [
                "class_names_temperature.joblib",
                "class_names.joblib"
            ]
        else:
            lgb_names = ["lgb_model.joblib"]
            cat_names = ["cat_model.joblib"]
            background_names = ["background_data.joblib"]
            class_names_files = ["class_names.joblib"]

        lgb_model, _ = load_first_existing(MODEL_DIR, lgb_names)
        cat_model, _ = load_first_existing(MODEL_DIR, cat_names)

        background_data, _ = load_first_existing(
            MODEL_DIR,
            background_names
        )

        try:
            class_names, _ = load_first_existing(
                MODEL_DIR,
                class_names_files
            )
            class_names = list(class_names)
        except FileNotFoundError:
            class_names = []

        if hasattr(lgb_model, "feature_names_in_"):
            actual = list(lgb_model.feature_names_in_)
            if actual != FEATURES:
                raise ValueError(
                    f"LightGBM feature mismatch. "
                    f"Metadata: {FEATURES}; model: {actual}"
                )

        if hasattr(cat_model, "feature_names_"):
            actual = list(cat_model.feature_names_)
            if actual and actual != FEATURES:
                raise ValueError(
                    f"CatBoost feature mismatch. "
                    f"Metadata: {FEATURES}; model: {actual}"
                )

        if hasattr(lgb_model, "n_features_in_"):
            if lgb_model.n_features_in_ != len(FEATURES):
                raise ValueError(
                    "LightGBM feature count does not match metadata."
                )

        # Validate CatBoost using its actual feature names.
        cat_feature_names = getattr(cat_model, "feature_names_", None)

        if cat_feature_names:
            if list(cat_feature_names) != FEATURES:
                raise ValueError(
                f"CatBoost feature mismatch. "
                f"Expected: {FEATURES}, "
                f"Found: {list(cat_feature_names)}"
                )
        else:
            cat_feature_count = getattr(cat_model, "n_features_in_", None)

            if cat_feature_count not in (None, 0, len(FEATURES)):
                raise ValueError(
                f"CatBoost expects {cat_feature_count} features, "
                f"but metadata contains {len(FEATURES)}."
                )

        print("Models loaded successfully.")
        print("Feature metadata:", feature_path.name)
        print("Features:", FEATURES)

        initialize_explainers()

        model_error = None
        return True

    except Exception as exc:
        model_error = str(exc)
        print("MODEL LOADING ERROR:", model_error)
        return False


def initialize_explainers():
    global shap_explainer
    global lime_explainer

    try:
        import shap

        shap_explainer = shap.TreeExplainer(lgb_model)

    except Exception as exc:
        shap_explainer = None
        print("SHAP unavailable:", exc)

    try:
        from lime.lime_tabular import LimeTabularExplainer

        lime_explainer = LimeTabularExplainer(
            np.asarray(background_data),
            feature_names=FEATURES,
            class_names=class_names or None,
            mode="classification",
            discretize_continuous=True,
            random_state=42
        )

    except Exception as exc:
        lime_explainer = None
        print("LIME unavailable:", exc)


load_models()

# ============================================================
# DATABASE
# ============================================================

def db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        str(DB_FILE),
        timeout=20,
        check_same_thread=False
    )

    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")

    return conn


def init_database():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with DB_LOCK:
        conn = db()

        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_id INTEGER,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                ph REAL NOT NULL,
                moisture REAL NOT NULL,
                temperature REAL,
                nitrogen REAL NOT NULL,
                cec REAL NOT NULL,
                fertility TEXT NOT NULL,
                confidence REAL NOT NULL,
                lgb_prediction TEXT NOT NULL,
                cat_prediction TEXT NOT NULL,
                lgb_confidence REAL NOT NULL,
                cat_confidence REAL NOT NULL,
                model_agreement INTEGER NOT NULL,
                shap_data TEXT,
                lime_data TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS app_state (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        conn.commit()
        conn.close()

    print("Database initialized.")


init_database()


# ============================================================
# PASSWORD HANDLING
# ============================================================

def hash_password(password):
    salt = secrets.token_bytes(16)

    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        200000
    )

    return salt.hex() + ":" + digest.hex()


def verify_password(password, stored):
    try:
        salt_hex, digest_hex = stored.split(":")

        candidate = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            200000
        )

        return secrets.compare_digest(
            candidate.hex(),
            digest_hex
        )

    except Exception:
        return False


def utc_now():
    return datetime.now(timezone.utc).isoformat()


# ============================================================
# LOCATION AND RASTER DATA
# ============================================================

RASTER_FILES = {
    "nitrogen": DATA_DIR / "nitrogen.tif",
    "cec": DATA_DIR / "cec.tif"
}


def get_location():
    with DB_LOCK:
        conn = db()

        row = conn.execute(
            "SELECT value FROM app_state WHERE key = ?",
            ("location",)
        ).fetchone()

        conn.close()

    if not row:
        return None

    try:
        return json.loads(row["value"])
    except Exception:
        return None


def raster_value(path, latitude, longitude):
    if not path.exists():
        raise FileNotFoundError(
            f"Required raster file is missing: {path.name}"
        )

    with rasterio.open(path) as src:

        if src.crs is None:
            raise ValueError(
                f"{path.name} does not contain a coordinate reference system."
            )

        xs, ys = warp_transform(
            "EPSG:4326",
            src.crs,
            [longitude],
            [latitude]
        )

        row, col = src.index(xs[0], ys[0])

        if (
            row < 0
            or col < 0
            or row >= src.height
            or col >= src.width
        ):
            raise ValueError(
                f"The selected location is outside {path.name}."
            )

        value = src.read(
            1,
            window=((row, row + 1), (col, col + 1))
        )[0, 0]

        if src.nodata is not None:
            if np.isclose(value, src.nodata):
                raise ValueError(
                    f"No raster data is available in {path.name} "
                    "for this location."
                )

        if not np.isfinite(value):
            raise ValueError(
                f"Invalid raster value in {path.name}."
            )

        return float(value)


def enrich_sensor_reading(ph, moisture, temperature, latitude, longitude):
    values = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "ph": float(ph),
        "moisture": float(moisture),
        "nitrogen": raster_value(
            RASTER_FILES["nitrogen"],
            latitude,
            longitude
        ),
        "cec": raster_value(
            RASTER_FILES["cec"],
            latitude,
            longitude
        )
    }

    if "temperature" in FEATURES:
        if temperature is None:
            raise ValueError(
                "The trained model requires temperature, "
                "but the Arduino did not provide it."
            )

        values["temperature"] = float(temperature)

    missing = [name for name in FEATURES if name not in values]

    if missing:
        raise ValueError(
            "Required model inputs are unavailable: "
            + ", ".join(missing)
        )

    return values


# ============================================================
# MACHINE LEARNING PREDICTION
# ============================================================

def predict_soil(values):
    if lgb_model is None or cat_model is None:
        raise RuntimeError(
            model_error or "The trained models are not available."
        )

    X = pd.DataFrame(
        [[values[name] for name in FEATURES]],
        columns=FEATURES
    )

    lgb_probs = np.asarray(
        lgb_model.predict_proba(X)
    )[0]

    cat_probs = np.asarray(
        cat_model.predict_proba(X)
    )[0]

    lgb_index = int(np.argmax(lgb_probs))
    cat_index = int(np.argmax(cat_probs))

    if hasattr(lgb_model, "classes_"):
        lgb_prediction = str(
            lgb_model.classes_[lgb_index]
        )
    else:
        lgb_prediction = str(
            lgb_model.predict(X)[0]
        )

    if hasattr(cat_model, "classes_"):
        cat_prediction = str(
            cat_model.classes_[cat_index]
        )
    elif class_names and cat_index < len(class_names):
        cat_prediction = str(class_names[cat_index])
    else:
        cat_prediction = str(cat_model.predict(X)[0])

    lgb_confidence = float(lgb_probs[lgb_index])
    cat_confidence = float(cat_probs[cat_index])

    agreement = lgb_prediction == cat_prediction

    if agreement:
        fertility = lgb_prediction
        confidence = (
            lgb_confidence + cat_confidence
        ) / 2
    elif lgb_confidence >= cat_confidence:
        fertility = lgb_prediction
        confidence = lgb_confidence
    else:
        fertility = cat_prediction
        confidence = cat_confidence

    result = {
        **values,
        "fertility": fertility,
        "confidence": float(confidence),
        "lgb_prediction": lgb_prediction,
        "cat_prediction": cat_prediction,
        "lgb_confidence": lgb_confidence,
        "cat_confidence": cat_confidence,
        "model_agreement": agreement,
        "timestamp": utc_now()
    }

    return result, X


# ============================================================
# SHAP AND LIME EXPLANATIONS
# ============================================================

def explain_prediction(X, result):
    shap_items = []
    lime_items = []

    if shap_explainer is not None:
        try:
            explanation = shap_explainer(X)
            values = np.asarray(explanation.values)

            if values.ndim == 3:
                class_index = 0

                if hasattr(lgb_model, "classes_"):
                    classes = list(lgb_model.classes_)

                    if result["lgb_prediction"] in classes:
                        class_index = classes.index(
                            result["lgb_prediction"]
                        )

                values = values[0, :, class_index]

            else:
                values = values[0]

            for feature, value in zip(FEATURES, values):
                shap_items.append({
                    "feature": feature,
                    "value": float(value)
                })

        except Exception as exc:
            print("SHAP explanation error:", exc)

    if lime_explainer is not None:
        try:
            explanation = lime_explainer.explain_instance(
                X.iloc[0].values,
                lambda array: lgb_model.predict_proba(
                    pd.DataFrame(
                        array,
                        columns=FEATURES
                    )
                ),
                num_features=len(FEATURES)
            )

            for feature, weight in explanation.as_list():
                lime_items.append({
                    "feature": feature,
                    "weight": float(weight)
                })

        except Exception as exc:
            print("LIME explanation error:", exc)

    return {
        "shap": shap_items,
        "lime": lime_items
    }


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_prediction(result, explanation):
    user_id = session.get("user_id")

    with DB_LOCK:
        conn = db()

        conn.execute("""
            INSERT INTO predictions (
                timestamp,
                user_id,
                latitude,
                longitude,
                ph,
                moisture,
                temperature,
                nitrogen,
                cec,
                fertility,
                confidence,
                lgb_prediction,
                cat_prediction,
                lgb_confidence,
                cat_confidence,
                model_agreement,
                shap_data,
                lime_data
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result["timestamp"],
            user_id,
            result["latitude"],
            result["longitude"],
            result["ph"],
            result["moisture"],
            result.get("temperature"),
            result["nitrogen"],
            result["cec"],
            result["fertility"],
            result["confidence"],
            result["lgb_prediction"],
            result["cat_prediction"],
            result["lgb_confidence"],
            result["cat_confidence"],
            int(result["model_agreement"]),
            json.dumps(explanation.get("shap", [])),
            json.dumps(explanation.get("lime", []))
        ))

        conn.commit()
        conn.close()


# ============================================================
# ARDUINO CONNECTION
# ============================================================

def connect_arduino():
    global arduino

    with arduino_lock:
        try:
            if arduino is not None and arduino.is_open:
                return True

            arduino = serial.Serial(
                SERIAL_PORT,
                SERIAL_BAUD,
                timeout=1
            )

            time.sleep(2)

            print(
                f"Arduino connected on {SERIAL_PORT} "
                f"at {SERIAL_BAUD} baud."
            )

            return True

        except Exception as exc:
            arduino = None
            print("Arduino connection error:", exc)
            return False


def read_arduino():
    global arduino

    if arduino is None:
        connect_arduino()
        return None

    try:
        line = arduino.readline().decode(
            "utf-8",
            errors="ignore"
        ).strip()

        if not line:
            return None

        # Accept comma-separated numeric readings:
        # pH,moisture
        # pH,moisture,temperature

        parts = line.split(",")

        if len(parts) not in (2, 3):
            return None

        values = [float(part.strip()) for part in parts]

        ph = values[0]
        moisture = values[1]
        temperature = values[2] if len(values) == 3 else None

        if not 0 <= ph <= 14:
            return None

        if not 0 <= moisture <= 100:
            return None

        if temperature is not None:
            if not -20 <= temperature <= 100:
                return None

        return ph, moisture, temperature

    except (ValueError, UnicodeDecodeError):
        return None

    except Exception as exc:
        print("Arduino read error:", exc)

        try:
            arduino.close()
        except Exception:
            pass

        arduino = None
        return None


# ============================================================
# SENSOR BACKGROUND THREAD
# ============================================================

def sensor_loop():
    global current

    while True:
        reading = read_arduino()

        if reading is None:
            with STATE_LOCK:
                current["connected"] = arduino is not None

                if arduino is None:
                    current["error"] = (
                        "Arduino disconnected. "
                        "No simulated sensor values are used."
                    )

            time.sleep(2)
            continue

        ph, moisture, temperature = reading

        with STATE_LOCK:
            current["connected"] = True
            current["reading"] = {
                "ph": ph,
                "moisture": moisture,
                "temperature": temperature
            }

        location_data = get_location()

        if not location_data:
            with STATE_LOCK:
                current["error"] = (
                    "Arduino is connected. "
                    "Set your field location to obtain "
                    "nitrogen and CEC raster values."
                )

            time.sleep(2)
            continue

        try:
            values = enrich_sensor_reading(
                ph,
                moisture,
                temperature,
                location_data["latitude"],
                location_data["longitude"]
            )

            result, X = predict_soil(values)

            explanation = explain_prediction(
                X,
                result
            )

            with STATE_LOCK:
                current = {
                    "connected": True,
                    "reading": values,
                    "prediction": result,
                    "explanation": explanation,
                    "timestamp": result["timestamp"],
                    "location_source": location_data.get(
                        "source",
                        "browser"
                    ),
                    "error": None
                }

        except Exception as exc:
            with STATE_LOCK:
                current["connected"] = True
                current["error"] = str(exc)

        time.sleep(2)


# ============================================================
# AUTHENTICATION API
# ============================================================

@app.route("/api/auth/signup", methods=["POST"])
@app.route("/api/signup", methods=["POST"])
def signup():
    data = request.get_json(silent=True) or {}

    name = str(data.get("name", "")).strip()
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))

    if not name or not email or len(password) < 6:
        return jsonify({
            "success": False,
            "error": (
                "Name, email and a password of at least "
                "6 characters are required."
            )
        }), 400

    try:
        with DB_LOCK:
            conn = db()

            cursor = conn.execute("""
                INSERT INTO users (
                    name,
                    email,
                    password_hash
                )
                VALUES (?, ?, ?)
            """, (
                name,
                email,
                hash_password(password)
            ))

            conn.commit()
            user_id = cursor.lastrowid
            conn.close()

        session["user_id"] = user_id
        session["user_name"] = name

        return jsonify({
            "success": True,
            "user": {
                "id": user_id,
                "name": name,
                "email": email
            }
        })

    except sqlite3.IntegrityError:
        return jsonify({
            "success": False,
            "error": "An account with that email already exists."
        }), 409


@app.route("/api/auth/login", methods=["POST"])
@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}

    email = str(
        data.get("email")
        or data.get("username")
        or data.get("identifier")
        or ""
    ).strip().lower()

    password = str(data.get("password", ""))

    if not email or not password:
        return jsonify({
            "success": False,
            "error": "Enter your email and password."
        }), 400

    with DB_LOCK:
        conn = db()

        row = conn.execute(
            "SELECT * FROM users WHERE lower(email) = ?",
            (email,)
        ).fetchone()

        conn.close()

    if not row or not verify_password(
        password,
        row["password_hash"]
    ):
        return jsonify({
            "success": False,
            "error": "Invalid email or password."
        }), 401

    session["user_id"] = row["id"]
    session["user_name"] = row["name"]

    return jsonify({
        "success": True,
        "logged_in": True,
        "user": {
            "id": row["id"],
            "name": row["name"],
            "email": row["email"]
        }
    })


@app.route("/api/auth/logout", methods=["POST"])
@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()

    return jsonify({
        "success": True
    })


@app.route("/api/auth/me", methods=["GET"])
@app.route("/api/me", methods=["GET"])
def me():
    if "user_id" not in session:
        return jsonify({
            "authenticated": False,
            "logged_in": False
        })

    with DB_LOCK:
        conn = db()

        row = conn.execute(
            "SELECT id, name, email FROM users WHERE id = ?",
            (session["user_id"],)
        ).fetchone()

        conn.close()

    return jsonify({
        "authenticated": bool(row),
        "logged_in": bool(row),
        "user": dict(row) if row else None
    })


# ============================================================
# LOCATION API
# ============================================================




# ============================================================
# LATEST SENSOR DATA AND STATUS
# ============================================================

@app.route("/api/latest", methods=["GET"])
def latest():
    with STATE_LOCK:
        return jsonify(current)


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "models_loaded": (
            lgb_model is not None
            and cat_model is not None
        ),
        "model_error": model_error,
        "arduino_connected": arduino is not None,
        "serial_port": SERIAL_PORT,
        "serial_baud": SERIAL_BAUD,
        "features": FEATURES,
        "shap_available": shap_explainer is not None,
        "lime_available": lime_explainer is not None,
        "real_data_only": True
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "real_data_only": True,
        "arduino_connected": arduino is not None,
        "models_loaded": (
            lgb_model is not None
            and cat_model is not None
        )
    })


# ============================================================
# MANUAL PREDICTION API
# ============================================================

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    try:
        latitude = float(data["latitude"])
        longitude = float(data["longitude"])
        ph = float(data["ph"])
        moisture = float(data["moisture"])

        temperature = data.get("temperature")

        if temperature is not None:
            temperature = float(temperature)

        if not -90 <= latitude <= 90:
            raise ValueError("Invalid latitude.")

        if not -180 <= longitude <= 180:
            raise ValueError("Invalid longitude.")

        if not 0 <= ph <= 14:
            raise ValueError("pH must be between 0 and 14.")

        if not 0 <= moisture <= 100:
            raise ValueError("Moisture must be between 0 and 100.")

        values = enrich_sensor_reading(
            ph,
            moisture,
            temperature,
            latitude,
            longitude
        )

        result, X = predict_soil(values)
        explanation = explain_prediction(X, result)

        save_prediction(result, explanation)

        with STATE_LOCK:
            current.update({
                "reading": values,
                "prediction": result,
                "explanation": explanation,
                "timestamp": result["timestamp"],
                "error": None
            })

        return jsonify({
            "success": True,
            "prediction": result,
            "explanation": explanation
        })

    except Exception as exc:
        return jsonify({
            "success": False,
            "error": str(exc)
        }), 400


# ============================================================
# HISTORY API
# ============================================================

@app.route("/api/history", methods=["GET"])
def history():
    with DB_LOCK:
        conn = db()

        rows = conn.execute("""
            SELECT
                id,
                timestamp,
                latitude,
                longitude,
                ph,
                moisture,
                temperature,
                nitrogen,
                cec,
                fertility,
                confidence,
                lgb_prediction,
                cat_prediction,
                lgb_confidence,
                cat_confidence,
                model_agreement
            FROM predictions
            ORDER BY id DESC
            LIMIT 100
        """).fetchall()

        conn.close()

    return jsonify([
        dict(row) for row in rows
    ])


# ============================================================
# PREDICTION EXPLANATION API
# ============================================================

@app.route(
    "/api/prediction/<int:prediction_id>/explanation",
    methods=["GET"]
)
def prediction_explanation(prediction_id):
    with DB_LOCK:
        conn = db()

        row = conn.execute("""
            SELECT
                shap_data,
                lime_data
            FROM predictions
            WHERE id = ?
        """, (prediction_id,)).fetchone()

        conn.close()

    if not row:
        return jsonify({
            "success": False,
            "error": "Prediction not found."
        }), 404

    try:
        shap_data = json.loads(row["shap_data"] or "[]")
        lime_data = json.loads(row["lime_data"] or "[]")
    except Exception:
        shap_data = []
        lime_data = []

    return jsonify({
        "success": True,
        "shap": shap_data,
        "lime": lime_data
    })


# ============================================================
# ANALYTICS API
# ============================================================

@app.route("/api/analytics", methods=["GET"])
def analytics():
    with DB_LOCK:
        conn = db()

        row = conn.execute("""
            SELECT
                COUNT(*) AS total,
                AVG(ph) AS avg_ph,
                AVG(moisture) AS avg_moisture,
                AVG(nitrogen) AS avg_nitrogen,
                AVG(cec) AS avg_cec,
                AVG(confidence) AS avg_confidence,
                AVG(model_agreement) AS agreement
            FROM predictions
        """).fetchone()

        distribution = conn.execute("""
            SELECT
                fertility,
                COUNT(*) AS count
            FROM predictions
            GROUP BY fertility
        """).fetchall()

        conn.close()

    return jsonify({
        "total_predictions": row["total"] or 0,
        "avg_ph": row["avg_ph"],
        "avg_moisture": row["avg_moisture"],
        "avg_nitrogen": row["avg_nitrogen"],
        "avg_cec": row["avg_cec"],
        "average_confidence": row["avg_confidence"],
        "model_agreement_rate": (
            row["agreement"] * 100
            if row["agreement"] is not None
            else None
        ),
        "fertility_distribution": [
            dict(item) for item in distribution
        ]
    })


# ============================================================
# WEATHER API
# ============================================================

@app.route("/api/weather", methods=["GET"])
def weather():
    location_data = get_location()

    if not location_data:
        return jsonify({
            "success": False,
            "error": "Set a field location first."
        }), 400

    params = urlencode({
        "latitude": location_data["latitude"],
        "longitude": location_data["longitude"],
        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "wind_speed_10m"
        ),
        "timezone": "auto"
    })

    try:
        url = (
            "https://api.open-meteo.com/v1/forecast?"
            + params
        )

        with urlopen(url, timeout=8) as response:
            data = json.loads(
                response.read().decode("utf-8")
            )

        return jsonify({
            "success": True,
            "location": location_data,
            "current": data.get("current", {})
        })

    except Exception as exc:
        return jsonify({
            "success": False,
            "error": f"Weather service unavailable: {exc}"
        }), 503


# ============================================================
# CROP RECOMMENDATION API
# ============================================================

@app.route("/api/crop-recommendations", methods=["GET"])
def crop_recommendations():
    with STATE_LOCK:
        reading = current.get("reading")

    if not reading or "nitrogen" not in reading:
        return jsonify({
            "success": False,
            "error": "Waiting for valid soil and location data."
        }), 409

    ph = reading["ph"]
    moisture = reading["moisture"]
    nitrogen = reading["nitrogen"]
    cec = reading["cec"]

    candidates = [
        (
            "Rice",
            6.0 <= ph <= 7.5 and moisture >= 55 and nitrogen >= 100,
            "Prefers moist soil and moderate nitrogen."
        ),
        (
            "Wheat",
            6.0 <= ph <= 7.5
            and 35 <= moisture <= 65
            and nitrogen >= 100,
            "Fits near-neutral pH with moderate moisture."
        ),
        (
            "Maize",
            5.8 <= ph <= 7.2
            and 35 <= moisture <= 70
            and nitrogen >= 120,
            "Benefits from adequate nitrogen and moderate moisture."
        ),
        (
            "Groundnut",
            5.5 <= ph <= 7.0
            and 25 <= moisture <= 60
            and nitrogen < 500,
            "Tolerates moderately acidic soil and moderate moisture."
        ),
        (
            "Millet",
            5.5 <= ph <= 7.5
            and 20 <= moisture <= 55,
            "Generally suited to lower moisture conditions."
        )
    ]

    recommendations = []

    for crop, suitable, reason in candidates:
        score = 0

        if suitable:
            score += 2

        if 6.0 <= ph <= 7.5:
            score += 1

        if 100 <= nitrogen <= 600:
            score += 1

        if cec >= 100:
            score += 1

        recommendations.append({
            "crop": crop,
            "score": score,
            "suitable": suitable,
            "reason": reason
        })

    recommendations.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return jsonify({
        "success": True,
        "recommendations": recommendations[:3],
        "basis": reading,
        "note": (
            "These are screening suggestions, not a substitute "
            "for local agronomic advice."
        )
    })


# ============================================================
# FRONTEND
# ============================================================



# ============================================================
# FRONTEND-COMPATIBILITY ENDPOINTS
# ============================================================

@app.route("/api/validate-location", methods=["POST"])
def validate_location():
    data = request.get_json(silent=True) or {}

    try:
        latitude = float(data["latitude"])
        longitude = float(data["longitude"])

        if not -90 <= latitude <= 90:
            raise ValueError("Latitude must be between -90 and 90.")
        if not -180 <= longitude <= 180:
            raise ValueError("Longitude must be between -180 and 180.")

        nitrogen = raster_value(
            RASTER_FILES["nitrogen"], latitude, longitude
        )
        cec = raster_value(
            RASTER_FILES["cec"], latitude, longitude
        )

        return jsonify({
            "success": True,
            "latitude": latitude,
            "longitude": longitude,
            "nitrogen": nitrogen,
            "cec": cec
        })

    except Exception as exc:
        return jsonify({
            "success": False,
            "error": str(exc)
        }), 400


@app.route("/api/find-high-test", methods=["GET"])
def find_high_test():
    # A labeled High-fertility test vector is not present in the
    # supplied CSV files. Do not fabricate one from unlabeled data.
    return jsonify({
        "success": False,
        "error": (
            "The supplied dataset does not contain a labeled "
            "High-fertility test vector. Add a verified labeled "
            "test record to enable this feature."
        )
    }), 404


@app.route("/")
def index():
    return send_from_directory(
        str(FRONTEND_DIR),
        "index.html"
    )


@app.route("/<path:path>")
def frontend_files(path):
    requested_file = FRONTEND_DIR / path

    if requested_file.is_file():
        return send_from_directory(
            str(FRONTEND_DIR),
            path
        )

    return send_from_directory(
        str(FRONTEND_DIR),
        "index.html"
    )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SOILAI - SOIL FERTILITY PREDICTION SYSTEM")
    print("=" * 60)

    print("Serial port:", SERIAL_PORT)
    print("Serial baud:", SERIAL_BAUD)
    print("Model features:", FEATURES)
    print("Models loaded:", lgb_model is not None and cat_model is not None)

    connect_arduino()

    threading.Thread(
        target=sensor_loop,
        daemon=True
    ).start()

    host = getattr(
        config,
        "FLASK_HOST",
        "127.0.0.1"
    )

    port = int(
        getattr(config, "FLASK_PORT", 5000)
    )

    print(f"Dashboard: http://localhost:{port}")

    app.run(
        host=host,
        port=port,
        debug=False,
        use_reloader=False
    )