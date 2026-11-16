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
from backend.advice_engine import SoilAdviceEngine

ADVICE_ENGINE = SoilAdviceEngine()

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

# Live state is reserved exclusively for real Arduino measurements.
current = {
    "connected": False,
    "reading": None,
    "error": "Waiting for Arduino data."
}
# Latest successful prediction inputs are separate from the live IoT panel.
last_prediction_values = None

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

        # Backward-compatible username migration. Existing users and hashes remain intact.
        user_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(users)").fetchall()
        }
        if "username" not in user_columns:
            conn.execute("ALTER TABLE users ADD COLUMN username TEXT")

        # Give legacy accounts a stable generated username so they remain addressable.
        legacy_users = conn.execute(
            "SELECT id, name FROM users WHERE username IS NULL OR username = ''"
        ).fetchall()
        for legacy_user in legacy_users:
            base = "".join(
                ch.lower() if ch.isalnum() else "_"
                for ch in str(legacy_user["name"]).strip()
            ).strip("_") or "farmer"
            generated = f"{base}_{legacy_user['id']}"
            conn.execute(
                "UPDATE users SET username = ? WHERE id = ?",
                (generated, legacy_user["id"])
            )

        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username_unique
            ON users(username COLLATE NOCASE)
            WHERE username IS NOT NULL AND username <> ''
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



def raster_value(path, latitude, longitude, max_radius=5):
    """
    Read a raster value at GPS coordinates.
    If the exact pixel is NoData, find the nearest valid
    pixel within max_radius cells.

    Returns:
        {
            "value": float or None,
            "distance_km": float or None,
            "estimated": bool
        }
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Raster file missing: {path.name}"
        )

    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(
                f"{path.name} has no coordinate reference system."
            )

        xs, ys = warp_transform(
            "EPSG:4326",
            src.crs,
            [float(longitude)],
            [float(latitude)]
        )

        x, y = xs[0], ys[0]
        row, col = src.index(x, y)

        if not (
            0 <= row < src.height
            and 0 <= col < src.width
        ):
            return {
                "value": None,
                "distance_km": None,
                "estimated": False
            }

        def read_pixel(r, c):
            pixel = src.read(
                1,
                window=((r, r + 1), (c, c + 1)),
                masked=True
            )[0, 0]

            if np.ma.is_masked(pixel):
                return None

            value = float(pixel)

            if not np.isfinite(value):
                return None

            if src.nodata is not None and np.isclose(
                value, src.nodata
            ):
                return None

            return value

        # First try the exact GPS pixel.
        exact_value = read_pixel(row, col)

        if exact_value is not None:
            return {
                "value": exact_value,
                "distance_km": 0.0,
                "estimated": False
            }

        # Search outward, stopping at the first radius
        # containing valid pixels.
        for radius in range(1, max_radius + 1):
            candidates = []

            for r in range(
                max(0, row - radius),
                min(src.height, row + radius + 1)
            ):
                for c in range(
                    max(0, col - radius),
                    min(src.width, col + radius + 1)
                ):
                    if max(abs(r - row), abs(c - col)) != radius:
                        continue

                    value = read_pixel(r, c)

                    if value is not None:
                        distance = (
                            ((r - row) * src.transform.e) ** 2
                            + ((c - col) * src.transform.a) ** 2
                        ) ** 0.5 / 1000.0

                        candidates.append(
                            (distance, value)
                        )

            if candidates:
                distance, value = min(
                    candidates,
                    key=lambda item: item[0]
                )

                return {
                    "value": value,
                    "distance_km": round(distance, 2),
                    "estimated": True
                }

        return {
            "value": None,
            "distance_km": None,
            "estimated": False
        }




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

        prediction_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.commit()
        conn.close()
        return prediction_id


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
    """Poll Arduino; history is written only on an explicit IoT prediction."""
    while True:
        reading = read_arduino()
        if reading is None:
            with STATE_LOCK:
                current["connected"] = False
                current["reading"] = None
                current["error"] = "Arduino disconnected. No simulated sensor values are used."
            time.sleep(2)
            continue
        ph, moisture, temperature = reading
        with STATE_LOCK:
            current["connected"] = True
            current["reading"] = {"ph": ph, "moisture": moisture, "temperature": temperature}
            current["error"] = None
        time.sleep(2)


# AUTHENTICATION API
# ============================================================

@app.route("/api/auth/signup", methods=["POST"])
@app.route("/api/signup", methods=["POST"])
def signup():
    data = request.get_json(silent=True) or {}

    name = str(data.get("name", "")).strip()
    username = str(data.get("username", "")).strip()
    contact = str(
        data.get("email") or data.get("contact") or data.get("mobile") or ""
    ).strip()
    password = str(data.get("password", ""))

    if not name or not username or not contact or len(password) < 6:
        return jsonify({
            "success": False,
            "error": "Name, username, email/mobile and a password of at least 6 characters are required."
        }), 400

    conn = None
    try:
        with DB_LOCK:
            conn = db()
            existing = conn.execute("""
                SELECT id FROM users
                WHERE lower(username) = lower(?) OR lower(email) = lower(?)
                LIMIT 1
            """, (username, contact)).fetchone()
            if existing:
                return jsonify({
                    "success": False,
                    "error": "That username or email/mobile is already registered."
                }), 409

            cursor = conn.execute("""
                INSERT INTO users (name, username, email, password_hash)
                VALUES (?, ?, ?, ?)
            """, (name, username, contact, hash_password(password)))
            conn.commit()
            user_id = cursor.lastrowid

        session["user_id"] = user_id
        session["user_name"] = name
        return jsonify({
            "success": True,
            "logged_in": True,
            "user": {
                "id": user_id,
                "name": name,
                "username": username,
                "email": contact
            }
        })
    except sqlite3.IntegrityError:
        if conn:
            conn.rollback()
        return jsonify({
            "success": False,
            "error": "That username or email/mobile is already registered."
        }), 409
    finally:
        if conn:
            conn.close()


@app.route("/api/auth/login", methods=["POST"])
@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    identifier = str(
        data.get("username") or data.get("identifier") or data.get("email") or ""
    ).strip()
    password = str(data.get("password", ""))

    if not identifier or not password:
        return jsonify({
            "success": False,
            "error": "Enter your username/email and password."
        }), 400

    with DB_LOCK:
        conn = db()
        row = conn.execute("""
            SELECT * FROM users
            WHERE lower(username) = lower(?) OR lower(email) = lower(?)
            LIMIT 1
        """, (identifier, identifier)).fetchone()
        conn.close()

    if not row or not verify_password(password, row["password_hash"]):
        return jsonify({
            "success": False,
            "error": "Invalid username/email or password."
        }), 401

    session["user_id"] = row["id"]
    session["user_name"] = row["name"]
    return jsonify({
        "success": True,
        "logged_in": True,
        "user": {
            "id": row["id"],
            "name": row["name"],
            "username": row["username"],
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
            "SELECT id, name, username, email FROM users WHERE id = ?",
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
        connected = bool(current.get("connected"))
        reading = dict(current.get("reading") or {}) if connected else {}
        error = current.get("error")
    location_data = get_location() if connected else None
    nitrogen = cec = None
    if location_data:
        nitrogen = raster_value(RASTER_FILES["nitrogen"], location_data["latitude"], location_data["longitude"])
        cec = raster_value(RASTER_FILES["cec"], location_data["latitude"], location_data["longitude"])
    return jsonify({
        "ph": reading.get("ph"), "moisture": reading.get("moisture"),
        "temperature": reading.get("temperature"), "nitrogen": nitrogen, "cec": cec,
        "latitude": location_data.get("latitude") if location_data else None,
        "longitude": location_data.get("longitude") if location_data else None,
        "connected": connected, "sensor_connected": connected, "error": error,
        "timestamp": utc_now() if connected and reading else None
    })


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

def build_crop_recommendations(soil_values):
    """Adapt the existing advice engine output for the frontend display."""
    ranked = ADVICE_ENGINE.recommend_crops(soil_values)
    recommendations = []
    for item in ranked:
        evaluated = int(item.get("evaluated_parameters", 0) or 0)
        matched = int(item.get("matched_parameters", 0) or 0)
        score = float(item.get("score", 0) or 0)
        recommendations.append({
            "crop": item.get("crop", "Crop"),
            "score": score,
            "suitability": f"{score:.0f}% suitability match",
            "matched_conditions": matched,
            "total_conditions": evaluated,
            "reason": item.get("reason", "Based on available soil measurements."),
            "conditions": {}
        })
    return recommendations


@app.route("/api/predict", methods=["POST"])
def predict():
    global last_prediction_values
    data = request.get_json(silent=True) or {}
    try:
        latitude, longitude = float(data["latitude"]), float(data["longitude"])
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ValueError("Latitude or longitude is outside its valid range.")
        input_mode = str(data.get("input_mode", "manual")).lower()
        if input_mode not in ("manual", "iot"):
            raise ValueError("input_mode must be 'manual' or 'iot'.")

        if input_mode == "iot":
            with STATE_LOCK:
                sensor_reading = dict(current.get("reading") or {})
                sensor_connected = bool(current.get("connected"))
            if not sensor_connected or not sensor_reading:
                raise ValueError("Arduino is not connected or has not provided a complete reading.")
            ph = float(sensor_reading["ph"])
            moisture = float(sensor_reading["moisture"])
            temperature = sensor_reading.get("temperature")
            values = enrich_sensor_reading(ph, moisture, temperature, latitude, longitude)
        else:
            ph, moisture = float(data["ph"]), float(data["moisture"])
            temperature = data.get("temperature")
            temperature = float(temperature) if temperature not in (None, "") else None
            nitrogen, cec = float(data["nitrogen"]), float(data["cec"])
            if not 0 <= ph <= 14: raise ValueError("pH must be between 0 and 14.")
            if not 0 <= moisture <= 100: raise ValueError("Moisture must be between 0 and 100.")
            if temperature is not None and not -50 <= temperature <= 70:
                raise ValueError("Temperature must be between -50 and 70 °C.")
            if nitrogen < 0 or cec < 0: raise ValueError("Nitrogen and CEC must be non-negative.")
            values = {"latitude": latitude, "longitude": longitude, "ph": ph,
                      "moisture": moisture, "nitrogen": nitrogen, "cec": cec}
            if temperature is not None: values["temperature"] = temperature

        missing = [name for name in FEATURES if name not in values or values[name] is None]
        if missing: raise ValueError("Required model inputs are missing: " + ", ".join(missing))
        result, X = predict_soil(values)
        explanation = explain_prediction(X, result)
        prediction_id = save_prediction(result, explanation)
        with STATE_LOCK:
            last_prediction_values = dict(values)

        crop_recommendations = build_crop_recommendations(values)
        advice = ADVICE_ENGINE.generate_advice(values)
        return jsonify({
            "success": True, "prediction_id": prediction_id, "input_mode": input_mode,
            "prediction": result, "explanation": explanation, "soil": values,
            "crop_recommendations": crop_recommendations,
            "crops": {"summary": "Crop suitability is screened using the soil measurements available for this prediction.",
                      "recommendations": crop_recommendations},
            "soil_advice": advice
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


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

    records = [dict(row) for row in rows]
    return jsonify({"success": True, "history": records, "predictions": records, "records": records})


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
    """Return crop screening for the latest successful manual or IoT prediction."""
    with STATE_LOCK:
        reading = dict(last_prediction_values) if last_prediction_values else None

    if not reading:
        with DB_LOCK:
            conn = db()
            row = conn.execute("""
                SELECT latitude, longitude, ph, moisture, temperature, nitrogen, cec
                FROM predictions
                ORDER BY id DESC
                LIMIT 1
            """).fetchone()
            conn.close()
        if row:
            reading = dict(row)

    if not reading:
        return jsonify({"success": False,
                        "error": "Run a soil prediction first to generate crop recommendations."}), 409
    recommendations = build_crop_recommendations(reading)
    advice = ADVICE_ENGINE.generate_advice(reading)
    return jsonify({
        "success": True, "recommendations": recommendations,
        "crop_recommendations": recommendations,
        "crops": {"recommendations": recommendations},
        "soil_advice": advice, "basis": reading,
        "note": advice.get("disclaimer")
    })


# FRONTEND-COMPATIBILITY ENDPOINTS
# ============================================================



@app.route("/api/location", methods=["GET", "POST"])
def location_api():
    if request.method == "GET":
        return jsonify({
            "success": True,
            "location": get_location()
        })

    data = request.get_json(silent=True) or {}

    try:
        latitude = float(data["latitude"])
        longitude = float(data["longitude"])

        if not -90 <= latitude <= 90:
            raise ValueError("Invalid latitude.")

        if not -180 <= longitude <= 180:
            raise ValueError("Invalid longitude.")

        location = {
            "latitude": latitude,
            "longitude": longitude,
            "source": str(data.get("source", "browser")),
            "updated_at": utc_now()
        }

        # Save GPS even if raster data is unavailable.
        with DB_LOCK:
            conn = db()
            conn.execute(
                "INSERT INTO app_state(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                ("location", json.dumps(location))
            )
            conn.commit()
            conn.close()

        raster_results = {}
        raster_errors = []

        for name, path in RASTER_FILES.items():
            try:
                result = raster_value(
                    path,
                    latitude,
                    longitude,
                    max_radius=5
                )

                raster_results[name] = result

                if result["value"] is None:
                    raster_errors.append(
                        f"No valid {name} data within the search distance."
                    )

            except Exception as exc:
                raster_results[name] = {
                    "value": None,
                    "distance_km": None,
                    "estimated": False
                }
                raster_errors.append(f"{name}: {exc}")

        nitrogen_result = raster_results.get(
            "nitrogen", {"value": None}
        )
        cec_result = raster_results.get(
            "cec", {"value": None}
        )

        nitrogen = nitrogen_result["value"]
        cec = cec_result["value"]

        raster_available = (
            nitrogen is not None and cec is not None
        )

        estimated = (
            nitrogen_result.get("estimated", False)
            or cec_result.get("estimated", False)
        )

        return jsonify({
            "success": True,
            "location": location,
            "latitude": latitude,
            "longitude": longitude,

            # Keep these numeric for frontend compatibility.
            "nitrogen": nitrogen,
            "cec": cec,

            "nitrogen_distance_km":
                nitrogen_result.get("distance_km"),
            "cec_distance_km":
                cec_result.get("distance_km"),

            "nitrogen_estimated":
                nitrogen_result.get("estimated", False),
            "cec_estimated":
                cec_result.get("estimated", False),

            "raster_available": raster_available,
            "estimated": estimated,

            "raster_message": (
                "GPS detected. Nearby soil raster values retrieved."
                if raster_available and estimated
                else "GPS detected. Exact soil raster values retrieved."
                if raster_available
                else "GPS detected, but one or more raster values "
                     "are unavailable. Manual entry is still available."
            ),

            "raster_errors": raster_errors
        })

    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({
            "success": False,
            "error": str(exc)
        }), 400



@app.route("/api/validate-location", methods=["POST"])
def validate_location():
    data = request.get_json(silent=True) or {}

    try:
        latitude = float(data["latitude"])
        longitude = float(data["longitude"])

        if not -90 <= latitude <= 90:
            raise ValueError("Invalid latitude.")

        if not -180 <= longitude <= 180:
            raise ValueError("Invalid longitude.")

        nitrogen_result = raster_value(
            RASTER_FILES["nitrogen"],
            latitude,
            longitude,
            max_radius=5
        )

        cec_result = raster_value(
            RASTER_FILES["cec"],
            latitude,
            longitude,
            max_radius=5
        )

        return jsonify({
            "success": True,
            "latitude": latitude,
            "longitude": longitude,

            "nitrogen": nitrogen_result["value"],
            "cec": cec_result["value"],

            "nitrogen_distance_km":
                nitrogen_result["distance_km"],
            "cec_distance_km":
                cec_result["distance_km"],

            "nitrogen_estimated":
                nitrogen_result["estimated"],
            "cec_estimated":
                cec_result["estimated"]
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