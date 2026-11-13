"use strict";

/* =========================================================
   SoilAI Frontend
========================================================= */

const $ = (id) => document.getElementById(id);

let currentPage = "overview";
let inputMode = "manual";
let currentLocation = null;
let latestPrediction = null;

let fertilityChart = null;
let sensorChart = null;

let latestHistory = [];
let appUIInitialized = false;


/* =========================================================
   HELPERS
========================================================= */

function showToast(message) {
    const toast = $("toast");
    if (!toast) return;

    toast.textContent = message;
    toast.classList.add("show");

    clearTimeout(window.toastTimer);

    window.toastTimer = setTimeout(() => {
        toast.classList.remove("show");
    }, 3000);
}

function formatNumber(value, digits = 2) {
    if (
        value === null ||
        value === undefined ||
        value === "" ||
        Number.isNaN(Number(value))
    ) {
        return "--";
    }

    return Number(value).toFixed(digits);
}

function formatPercent(value) {
    if (
        value === null ||
        value === undefined ||
        value === "" ||
        Number.isNaN(Number(value))
    ) {
        return "--";
    }

    const number = Number(value);

    if (number <= 1) {
        return `${(number * 100).toFixed(1)}%`;
    }

    return `${number.toFixed(1)}%`;
}

function escapeHTML(value) {
    if (value === null || value === undefined) {
        return "";
    }

    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

async function getJSON(url, options = {}) {
    const response = await fetch(url, {
        credentials: "same-origin",
        ...options,
        headers: {
            "Content-Type": "application/json",
            ...(options.headers || {})
        }
    });

    let data = {};

    try {
        data = await response.json();
    } catch {
        data = {};
    }

    if (!response.ok) {
        throw new Error(
            data.error ||
            data.message ||
            `Request failed (${response.status})`
        );
    }

    return data;
}


/* =========================================================
   AUTH TABS
========================================================= */

function setupAuthTabs() {
    document.querySelectorAll(".auth-tab").forEach((button) => {
        button.addEventListener("click", () => {
            const mode = button.dataset.auth;

            document.querySelectorAll(".auth-tab")
                .forEach((item) => item.classList.remove("active"));

            button.classList.add("active");

            $("loginForm").classList.toggle(
                "hidden",
                mode !== "login"
            );

            $("signupForm").classList.toggle(
                "hidden",
                mode !== "signup"
            );
        });
    });
}


/* =========================================================
   LOGIN
========================================================= */

async function login(event) {
    event.preventDefault();

    const username = $("loginUsername").value.trim();
    const password = $("loginPassword").value;

    if (!username || !password) {
        showToast("Enter your username and password.");
        return;
    }

    try {
        const data = await getJSON("/api/login", {
            method: "POST",
            body: JSON.stringify({
                username,
                password
            })
        });

        if (data.success === false) {
            throw new Error(data.error || "Login failed.");
        }

        localStorage.setItem(
            "soilai_user",
            JSON.stringify(data.user || { username })
        );

        showApp(data.user || { username });

        showToast("Login successful.");

    } catch (error) {
        showToast(error.message);
    }
}


/* =========================================================
   SIGNUP
========================================================= */

async function signup(event) {
    event.preventDefault();

    const name = $("signupName").value.trim();
    const username = $("signupUsername").value.trim();
    const contact = $("signupContact").value.trim();
    const password = $("signupPassword").value;

    if (!name || !username || !contact || !password) {
        showToast("Please complete all fields.");
        return;
    }

    try {
        const data = await getJSON("/api/signup", {
            method: "POST",
            body: JSON.stringify({
                name,
                username,
                email: contact,
                mobile: contact,
                contact,
                password
            })
        });

        if (data.success === false) {
            throw new Error(data.error || "Signup failed.");
        }

        showToast("Account created. You can now log in.");

        $("signupForm").reset();

        document.querySelector('[data-auth="login"]').click();

        $("loginUsername").value = username;

    } catch (error) {
        showToast(error.message);
    }
}


/* =========================================================
   SHOW APP
========================================================= */

function showApp(user = {}) {
    $("authScreen").classList.add("hidden");
    $("appScreen").classList.remove("hidden");

    const name = user.name || user.username || "Farmer";
    const firstLetter = name.trim().charAt(0).toUpperCase() || "F";

    $("userAvatar").textContent = firstLetter;

    if (!appUIInitialized) {
        setupNavigation();
        setupInputMode();
        setupLocation();
        setupTheme();
        appUIInitialized = true;
    }

    refreshAll();
}


/* =========================================================
   LOGOUT
========================================================= */

async function logout() {
    try {
        await getJSON("/api/logout", {
            method: "POST"
        });
    } catch {
        // Local fallback if backend endpoint is unavailable.
    }

    localStorage.removeItem("soilai_user");

    $("appScreen").classList.add("hidden");
    $("authScreen").classList.remove("hidden");

    showToast("Logged out.");
}


/* =========================================================
   NAVIGATION
========================================================= */

function setupNavigation() {
    document.querySelectorAll(".nav-item").forEach((button) => {
        button.addEventListener("click", () => {
            const page = button.dataset.page;
            if (!page) return;
            switchPage(page);
        });
    });
}

function switchPage(page) {
    currentPage = page;

    document.querySelectorAll(".nav-item")
        .forEach((button) => {
            button.classList.toggle(
                "active",
                button.dataset.page === page
            );
        });

    document.querySelectorAll(".page")
        .forEach((section) => {
            section.classList.toggle(
                "active-page",
                section.id === `page-${page}`
            );
        });

    const titles = {
        overview: "Overview",
        analytics: "Analytics",
        explainability: "Explainability",
        crops: "Crop recommendations",
        history: "Prediction history"
    };

    $("pageTitle").textContent = titles[page] || "Overview";

    if (page === "analytics") loadAnalytics();
    if (page === "history") loadHistory();
    if (page === "explainability") renderExplainability(latestPrediction);
    if (page === "crops") renderCrops(latestPrediction);
}


/* =========================================================
   INPUT MODE
========================================================= */

function setupInputMode() {
    $("manualModeBtn").addEventListener(
        "click",
        () => setInputMode("manual")
    );

    $("iotModeBtn").addEventListener(
        "click",
        () => setInputMode("iot")
    );

    $("predictManualBtn").addEventListener(
        "click",
        submitManualPrediction
    );

    $("refreshSensorBtn").addEventListener(
        "click",
        refreshLatest
    );

    $("predictIotBtn")?.addEventListener(
        "click",
        submitIoTPrediction
    );
}

function setInputMode(mode) {
    inputMode = mode;

    const manual = mode === "manual";

    $("manualModeBtn").classList.toggle("active", manual);
    $("iotModeBtn").classList.toggle("active", !manual);

    $("manualInputPanel").classList.toggle("hidden", !manual);
    $("iotInputPanel").classList.toggle("hidden", manual);
}


/* =========================================================
   LOCATION
========================================================= */

async function setupLocation() {
    $("useLocationBtn").addEventListener(
        "click",
        requestLocation
    );

    $("validateLocationBtn").addEventListener(
        "click",
        validateManualLocation
    );

    $("findHighTestBtn").addEventListener(
        "click",
        findHighTest
    );

    await loadSavedLocation();
}

function requestLocation() {
    if (!navigator.geolocation) {
        showToast("Location is not supported by this browser.");
        return;
    }

    navigator.geolocation.getCurrentPosition(
        (position) => {
            currentLocation = {
                latitude: position.coords.latitude,
                longitude: position.coords.longitude
            };

            setManualLocationFields();
            updateLocationDisplay();

            getJSON("/api/location", {
                method: "POST",
                body: JSON.stringify({
                    latitude: currentLocation.latitude,
                    longitude: currentLocation.longitude,
                    source: "browser"
                })
            }).catch(() => {});

            showToast("Location updated.");
        },
        () => {
            showToast("Location permission was not available.");
        },
        {
            enableHighAccuracy: true,
            timeout: 10000,
            maximumAge: 60000
        }
    );
}

async function loadSavedLocation() {
    try {
        const data = await getJSON("/api/location");
        const location = data.location;

        if (
            location &&
            Number.isFinite(Number(location.latitude)) &&
            Number.isFinite(Number(location.longitude))
        ) {
            currentLocation = {
                latitude: Number(location.latitude),
                longitude: Number(location.longitude)
            };

            setManualLocationFields();
            updateLocationDisplay();
        }
    } catch {
        // Manual coordinates can still be entered.
    }
}

function setManualLocationFields() {
    if (!currentLocation) return;

    const lat = $("manualLatitude");
    const lon = $("manualLongitude");

    if (lat) lat.value = Number(currentLocation.latitude).toFixed(6);
    if (lon) lon.value = Number(currentLocation.longitude).toFixed(6);
}

function getManualLocation() {
    const latText = $("manualLatitude").value.trim();
    const lonText = $("manualLongitude").value.trim();

    if (latText === "" || lonText === "") {
        throw new Error("Enter latitude and longitude.");
    }

    const latitude = Number(latText);
    const longitude = Number(lonText);

    if (!Number.isFinite(latitude) || latitude < -90 || latitude > 90) {
        throw new Error("Enter a valid latitude between -90 and 90.");
    }

    if (!Number.isFinite(longitude) || longitude < -180 || longitude > 180) {
        throw new Error("Enter a valid longitude between -180 and 180.");
    }

    return { latitude, longitude };
}

async function validateManualLocation() {
    const button = $("validateLocationBtn");

    try {
        const location = getManualLocation();

        button.disabled = true;
        button.textContent = "Checking...";

        const data = await getJSON("/api/validate-location", {
            method: "POST",
            body: JSON.stringify(location)
        });

        currentLocation = location;
        setManualLocationFields();
        updateLocationDisplay();

        if (data.nitrogen !== undefined) {
            $("nitrogenValue").textContent =
                formatNumber(data.nitrogen, 2);
        }

        if (data.cec !== undefined) {
            $("cecValue").textContent =
                formatNumber(data.cec, 2);
        }

        await getJSON("/api/location", {
            method: "POST",
            body: JSON.stringify({
                ...location,
                source: "manual"
            })
        });

        showToast("Soil data is available at this location.");

    } catch (error) {
        showToast(error.message || "Location validation failed.");

    } finally {
        button.disabled = false;
        button.textContent = "Check soil data";
    }
}

function updateLocationDisplay() {
    if (!currentLocation) {
        $("locationText").textContent = "Location unavailable";
        $("coordinatesValue").textContent = "--";
        return;
    }

    const lat = Number(currentLocation.latitude);
    const lon = Number(currentLocation.longitude);

    $("locationText").textContent =
        `${lat.toFixed(4)}, ${lon.toFixed(4)}`;

    $("coordinatesValue").innerHTML =
        `${lat.toFixed(4)}<br>${lon.toFixed(4)}`;
}
/* =========================================================
   HIGH-FERTILITY TEST INPUT
========================================================= */

async function findHighTest() {
    const button = $("findHighTestBtn");

    if (!button) return;

    button.disabled = true;
    button.textContent = "Searching...";

    try {
        const data = await getJSON("/api/find-high-test");

        if (!data.success || !data.test) {
            throw new Error(
                data.error || "Could not find a High test vector."
            );
        }

        const test = data.test;

        $("manualLatitude").value =
            Number(test.latitude).toFixed(6);

        $("manualLongitude").value =
            Number(test.longitude).toFixed(6);

        $("manualPh").value =
            Number(test.ph).toFixed(1);

        $("manualMoisture").value =
            Number(test.moisture).toFixed(1);

        $("manualTemperature").value =
            Number(test.temperature).toFixed(1);

        $("manualNitrogen").value =
            Number(test.nitrogen).toFixed(1);

        $("manualCec").value =
            Number(test.cec).toFixed(1);

        showToast(
            "High test found. Click Predict fertility."
        );

    } catch (error) {
        showToast(error.message);

    } finally {
        button.disabled = false;
        button.textContent = "Find High test";
    }
}


/* =========================================================
   MANUAL PREDICTION
========================================================= */

async function submitManualPrediction() {
    const ph = $("manualPh").value;
    const moisture = $("manualMoisture").value;
    const temperature = $("manualTemperature").value;
    const nitrogen = $("manualNitrogen").value;
    const cec = $("manualCec").value;

    if (
        ph === "" ||
        moisture === "" ||
        temperature === "" ||
        nitrogen === "" ||
        cec === ""
    ) {
        showToast(
            "Enter pH, moisture, temperature, nitrogen and CEC."
        );
        return;
    }

    let location;

    try {
        location = getManualLocation();
    } catch (error) {
        showToast(error.message);
        return;
    }

    currentLocation = location;
    updateLocationDisplay();

    const values = {
        ph: Number(ph),
        moisture: Number(moisture),
        temperature: Number(temperature),
        nitrogen: Number(nitrogen),
        cec: Number(cec)
    };

    if (
        !Number.isFinite(values.ph) ||
        values.ph < 0 ||
        values.ph > 14
    ) {
        showToast("Enter a valid pH value.");
        return;
    }

    if (
        !Number.isFinite(values.moisture) ||
        values.moisture < 0 ||
        values.moisture > 100
    ) {
        showToast("Enter moisture between 0 and 100%.");
        return;
    }

    if (
        !Number.isFinite(values.temperature) ||
        values.temperature < -50 ||
        values.temperature > 70
    ) {
        showToast("Enter a valid temperature.");
        return;
    }

    if (
        !Number.isFinite(values.nitrogen) ||
        values.nitrogen < 0 ||
        values.nitrogen > 2000
    ) {
        showToast("Enter nitrogen between 0 and 2000.");
        return;
    }

    if (
        !Number.isFinite(values.cec) ||
        values.cec < 0 ||
        values.cec > 1000
    ) {
        showToast("Enter CEC between 0 and 1000.");
        return;
    }

    const button = $("predictManualBtn");

    button.disabled = true;
    button.textContent = "Predicting...";

    try {
        const data = await getJSON("/api/predict", {
            method: "POST",
            body: JSON.stringify({
                input_mode: "manual",
                latitude: location.latitude,
                longitude: location.longitude,
                ph: values.ph,
                moisture: values.moisture,
                temperature: values.temperature,
                nitrogen: values.nitrogen,
                cec: values.cec
            })
        });

        if (data.success === false) {
            throw new Error(
                data.error || "Prediction failed."
            );
        }

        handlePrediction(data);

        if (data.prediction_id) {
            loadPredictionExplanation(data.prediction_id);
        }

        showToast("Soil fertility prediction completed.");

    } catch (error) {
        showToast(error.message);

    } finally {
        button.disabled = false;
        button.textContent = "Predict fertility";
    }
}


/* =========================================================
   IOT PREDICTION
========================================================= */

async function submitIoTPrediction() {
    const button = $("predictIotBtn");
    if (button) { button.disabled = true; button.textContent = "Predicting..."; }

    try {
        if (!currentLocation) {
            throw new Error("Set your field location before predicting.");
        }

        const sensor = await getJSON("/api/latest");
        if (!(sensor.sensor_connected === true || sensor.connected === true)) {
            throw new Error("Arduino sensor is not connected.");
        }
        if (sensor.ph == null || sensor.moisture == null || sensor.temperature == null) {
            throw new Error("Waiting for complete pH, moisture and temperature readings.");
        }

        const data = await getJSON("/api/predict", {
            method: "POST",
            body: JSON.stringify({
                input_mode: "iot",
                latitude: currentLocation.latitude,
                longitude: currentLocation.longitude
            })
        });

        if (data.success === false) throw new Error(data.error || "Prediction failed.");
        handlePrediction(data);
        if (data.prediction_id != null) loadPredictionExplanation(data.prediction_id);
        showToast("IoT soil fertility prediction completed.");
    } catch (error) {
        showToast(error.message);
    } finally {
        if (button) { button.disabled = false; button.textContent = "Predict using IoT"; }
    }
}


/* =========================================================
   SENSOR DATA - PH, MOISTURE AND TEMPERATURE
========================================================= */

// Display all three sensor values in the IoT sensing panel.
async function refreshLatest() {
    try {
        const data = await getJSON("/api/latest");

        if (!data) return;

        updateSensorStatus(data);

        // -----------------------------------------------
        // PH SENSOR
        // -----------------------------------------------

        if (
            data.ph !== undefined &&
            data.ph !== null &&
            data.ph !== ""
        ) {
            const ph = formatNumber(data.ph, 2);

            if ($("iotPH")) {
                $("iotPH").textContent = ph;
            }

            if ($("phValue")) {
                $("phValue").textContent = ph;
            }
        }

        // -----------------------------------------------
        // MOISTURE SENSOR
        // -----------------------------------------------

        if (
            data.moisture !== undefined &&
            data.moisture !== null &&
            data.moisture !== ""
        ) {
            const moisture = formatNumber(
                data.moisture,
                1
            );

            if ($("iotMoisture")) {
                $("iotMoisture").textContent = moisture;
            }

            if ($("moistureValue")) {
                $("moistureValue").textContent =
                    `${moisture} %`;
            }
        }

        // -----------------------------------------------
        // TEMPERATURE SENSOR
        // -----------------------------------------------

        if (
            data.temperature !== undefined &&
            data.temperature !== null &&
            data.temperature !== ""
        ) {
            const temperature = formatNumber(
                data.temperature,
                1
            );

            if ($("iotTemperature")) {
                $("iotTemperature").textContent =
                    temperature;
            }

            if ($("temperatureValue")) {
                $("temperatureValue").textContent =
                    `${temperature} °C`;
            }
        }

        // -----------------------------------------------
        // NITROGEN
        // -----------------------------------------------

        if (
            data.nitrogen !== undefined &&
            data.nitrogen !== null
        ) {
            $("nitrogenValue").textContent =
                formatNumber(data.nitrogen, 2);
        }

        // -----------------------------------------------
        // CEC
        // -----------------------------------------------

        if (
            data.cec !== undefined &&
            data.cec !== null
        ) {
            $("cecValue").textContent =
                formatNumber(data.cec, 2);
        }

        // -----------------------------------------------
        // LOCATION
        // -----------------------------------------------

        if (
            data.latitude !== undefined &&
            data.latitude !== null &&
            data.longitude !== undefined &&
            data.longitude !== null
        ) {
            currentLocation = {
                latitude: Number(data.latitude),
                longitude: Number(data.longitude)
            };

            updateLocationDisplay();
        }

        // -----------------------------------------------
        // PREDICTION
        // -----------------------------------------------

        if (data.fertility) {
            handlePrediction(data);
        }

    } catch (error) {
        console.error("Sensor refresh error:", error);
        updateConnection(false);
    }
}


/* =========================================================
   SENSOR CONNECTION STATUS
========================================================= */

function updateSensorStatus(data) {
    const connected =
        data.sensor_connected === true ||
        data.connected === true;

    $("iotStatusText").textContent =
        connected
            ? "Sensor connected"
            : "Waiting for sensor";

    $("iotStatusSubtext").textContent =
        connected
            ? "Receiving current field measurements"
            : "Connect the Arduino to begin receiving data";

    $("connectionText").textContent =
        connected
            ? "System connected"
            : "Backend connected";

    $("connectionSubtext").textContent =
        connected
            ? "IoT sensor available"
            : "Waiting for IoT data";

    $("connectionText").parentElement
        .previousElementSibling
        ?.classList.toggle("offline", !connected);
}


/* =========================================================
   BACKGROUND EXPLANATION
========================================================= */

async function loadPredictionExplanation(predictionId) {
    for (let attempt = 0; attempt < 20; attempt++) {
        try {
            const data = await getJSON(
                `/api/prediction/${encodeURIComponent(predictionId)}/explanation`
            );

            if (data.ready) {
                latestPrediction = {
                    ...latestPrediction,
                    shap: data.shap || [],
                    lime: data.lime || []
                };

                renderExplainability(latestPrediction);
                return;
            }

        } catch {
            return;
        }

        await new Promise(resolve =>
            setTimeout(resolve, 300)
        );
    }
}


/* =========================================================
   PREDICTION DISPLAY
========================================================= */

function handlePrediction(data) {
    latestPrediction = data;

    const soil =
        data?.soil ||
        data?.inputs ||
        data?.input ||
        data?.prediction_details ||
        data?.prediction_data ||
        data ||
        {};

    const models = data?.models || {};
    const lgb = models.lightgbm || {};
    const cat = models.catboost || {};

    const fertility =
        data?.fertility ||
        data?.prediction ||
        data?.fertility_class ||
        null;

    if (!fertility) return;

    const normalized = String(fertility).toLowerCase();

    $("fertilityValue").textContent = String(fertility);
    $("fertilityBadge").textContent =
        String(fertility).toUpperCase();

    $("fertilityBadge").className =
        `fertility-badge ${getFertilityClass(normalized)}`;

    $("fertilityMessage").textContent =
        getFertilityMessage(normalized);

    $("confidenceValue").textContent =
        formatPercent(
            data?.confidence ??
            data?.fertility_confidence
        );

    const agreement =
        data?.agreement ??
        data?.model_agreement ??
        data?.models?.agreement ??
        data?.prediction?.model_agreement ??
        data?.prediction?.agreement;

    $("agreementValue").textContent =
        agreement === true
            ? "Yes"
            : agreement === false
                ? "No"
                : "--";

    $("lastUpdate").textContent =
        formatTimestamp(
            data?.timestamp ||
            data?.created_at ||
            new Date().toISOString()
        );

    $("lgbPrediction").textContent =
        data?.lgb_prediction ||
        data?.lightgbm_prediction ||
        lgb.prediction ||
        "--";

    $("catPrediction").textContent =
        data?.cat_prediction ||
        data?.catboost_prediction ||
        cat.prediction ||
        "--";

    $("lgbConfidence").textContent =
        formatPercent(
            data?.lgb_confidence ?? lgb.confidence
        );

    $("catConfidence").textContent =
        formatPercent(
            data?.cat_confidence ?? cat.confidence
        );

    const ph =
        data?.ph ??
        data?.soil?.ph ??
        data?.inputs?.ph ??
        data?.input?.ph;

    const moisture =
        data?.moisture ??
        data?.soil?.moisture ??
        data?.inputs?.moisture ??
        data?.input?.moisture;

    const temperature =
        data?.temperature ??
        data?.soil?.temperature ??
        data?.inputs?.temperature ??
        data?.input?.temperature;

    if (ph !== undefined && ph !== null) {
        $("phValue").textContent =
            formatNumber(ph, 2);
    }

    if (moisture !== undefined && moisture !== null) {
        $("moistureValue").textContent =
            `${formatNumber(moisture, 1)} %`;
    }

    if (temperature !== undefined && temperature !== null) {
        $("temperatureValue").textContent =
            `${formatNumber(temperature, 1)} °C`;
    }

    if (soil.nitrogen !== undefined) {
        $("nitrogenValue").textContent =
            formatNumber(soil.nitrogen, 2);
    }

    if (soil.cec !== undefined) {
        $("cecValue").textContent =
            formatNumber(soil.cec, 2);
    }

    if (data?.shap || data?.lime) {
        renderExplainability(data);
    }

    renderCrops(data);
}


function getFertilityClass(value) {
    if (value.includes("high")) return "high";
    if (value.includes("medium")) return "medium";
    if (value.includes("low")) return "low";
    return "neutral";
}

function getFertilityMessage(value) {
    if (value.includes("high")) {
        return "Soil conditions are favourable.";
    }

    if (value.includes("medium")) {
        return "Soil conditions may benefit from management.";
    }

    if (value.includes("low")) {
        return "Soil conditions may require improvement.";
    }

    return "Prediction available.";
}
/* =========================================================
   EXPLAINABILITY
========================================================= */

function renderExplainability(data) {
    const ex = data || {};

    const shap = Array.isArray(ex.shap) ? ex.shap : [];
    const lime = Array.isArray(ex.lime) ? ex.lime : [];

    if (!shap.length) {
        $("shapList").innerHTML =
            '<div class="empty">SHAP is waiting for a valid explanation.</div>';
    } else {
        const sorted = [...shap].sort(
            (a, b) =>
                Math.abs(Number(b.value ?? b.contribution ?? 0)) -
                Math.abs(Number(a.value ?? a.contribution ?? 0))
        );

        const max = Math.max(
            ...sorted.map(item =>
                Math.abs(Number(item.value ?? item.contribution ?? 0))
            ),
            1
        );

        $("shapList").innerHTML = sorted.map(item => {
            const feature = item.feature || item.name || "Feature";
            const value = Number(item.value ?? item.contribution ?? 0);
            const width = Math.min(100, Math.abs(value) / max * 100);
            const negative = value < 0 ? "negative" : "";

            return `
                <div class="explanation-item">
                    <div class="explanation-item-top">
                        <span class="explanation-feature">
                            ${escapeHTML(feature)}
                        </span>
                        <span class="explanation-value">
                            ${value >= 0 ? "+" : ""}${value.toFixed(4)}
                        </span>
                    </div>
                    <div class="explanation-bar">
                        <div class="explanation-fill ${negative}"
                             style="width:${width}%"></div>
                    </div>
                </div>
            `;
        }).join("");
    }

    if (!lime.length) {
        $("limeList").innerHTML =
            '<div class="empty">LIME is waiting for a valid explanation.</div>';
    } else {
        $("limeList").innerHTML = lime.map(item => {
            const feature =
                item.feature || item.name || item.rule || "Feature";

            const value = Number(
                item.value ?? item.weight ?? item.contribution ?? 0
            );

            return `
                <div class="explanation-item">
                    <div class="explanation-item-top">
                        <span class="explanation-feature">
                            ${escapeHTML(feature)}
                        </span>
                        <span class="explanation-value">
                            ${value >= 0 ? "+" : ""}${value.toFixed(4)}
                        </span>
                    </div>
                </div>
            `;
        }).join("");
    }
}


/* =========================================================
   CROP RECOMMENDATIONS
========================================================= */

function renderCrops(data) {
    const container = $("cropRecommendation");

    if (!container) return;

    const cropData = data?.crops || {};

    const recommendations =
        cropData.recommendations ||
        data?.recommendations ||
        data?.crop_recommendations ||
        [];

    if (
        !Array.isArray(recommendations) ||
        recommendations.length === 0
    ) {
        container.innerHTML = `
            <div class="empty">
                No crop recommendations available.
            </div>
        `;
        return;
    }

    const summary =
        cropData.summary ||
        "Crop suitability is evaluated from current soil conditions.";

    container.innerHTML = `
        <div class="crop-summary">
            ${escapeHTML(summary)}
        </div>

        <div class="crop-list">
            ${recommendations.map((item, index) => {
                const crop = item.crop || "Recommended crop";
                const suitability = item.suitability || "Recommended";
                const reason =
                    item.reason || "Based on current soil conditions.";

                const matched = item.matched_conditions ?? 0;
                const total = item.total_conditions ?? 3;
                const conditions = item.conditions || {};

                const conditionHTML =
                    Object.entries(conditions).map(([name, passed]) => `
                        <span class="condition ${passed ? "pass" : "fail"}">
                            ${passed ? "✓" : "✗"}
                            ${escapeHTML(name)}
                        </span>
                    `).join("");

                return `
                    <div class="crop-card">
                        <div class="crop-recommendation-header">
                            <div>
                                <h3>
                                    ${index + 1}. 🌱 ${escapeHTML(crop)}
                                </h3>
                                <span class="crop-suitability">
                                    ${escapeHTML(suitability)}
                                </span>
                            </div>

                            <span class="crop-score">
                                ${matched}/${total} conditions
                            </span>
                        </div>

                        <div class="crop-conditions">
                            ${conditionHTML}
                        </div>

                        <div class="crop-recommendation-reason">
                            <strong>🔍 Why?</strong>
                            <p>${escapeHTML(reason)}</p>
                        </div>
                    </div>
                `;
            }).join("")}
        </div>
    `;
}


/* =========================================================
   HISTORY
========================================================= */

async function loadHistory() {
    try {
        const data = await getJSON("/api/history");

        const rows =
            data.history ||
            data.predictions ||
            data.records ||
            [];

        latestHistory = Array.isArray(rows) ? rows : [];

        renderHistory(latestHistory);

    } catch (error) {
        $("historyBody").innerHTML = `
            <tr>
                <td colspan="7" class="empty-cell">
                    Unable to load prediction history.
                </td>
            </tr>
        `;
    }
}


function renderHistory(rows) {
    if (!rows.length) {
        $("historyBody").innerHTML = `
            <tr>
                <td colspan="7" class="empty-cell">
                    No prediction records yet.
                </td>
            </tr>
        `;
        return;
    }

    $("historyBody").innerHTML = rows.map(row => {
        const fertility =
            row.fertility || row.prediction || "--";

        return `
            <tr>
                <td>
                    ${escapeHTML(formatTimestamp(
                        row.timestamp || row.created_at
                    ))}
                </td>

                <td>
                    ${escapeHTML(formatCoordinates(
                        row.latitude,
                        row.longitude
                    ))}
                </td>

                <td>
                    ${escapeHTML(formatNumber(row.ph, 2))}
                </td>

                <td>
                    ${escapeHTML(formatNumber(row.moisture, 1))}%
                </td>

                <td>
                    ${
                        row.temperature === null ||
                        row.temperature === undefined
                            ? "--"
                            : `${escapeHTML(
                                formatNumber(row.temperature, 1)
                            )}°C`
                    }
                </td>

                <td>${escapeHTML(fertility)}</td>

                <td>
                    ${escapeHTML(formatPercent(row.confidence))}
                </td>
            </tr>
        `;
    }).join("");
}


function formatCoordinates(lat, lon) {
    if (
        lat === undefined ||
        lat === null ||
        lon === undefined ||
        lon === null
    ) {
        return "--";
    }

    return `${Number(lat).toFixed(3)}, ${Number(lon).toFixed(3)}`;
}


function formatTimestamp(value) {
    if (!value) return "--";

    const date = new Date(value);

    if (Number.isNaN(date.getTime())) {
        return String(value);
    }

    return date.toLocaleString([], {
        day: "2-digit",
        month: "short",
        hour: "2-digit",
        minute: "2-digit"
    });
}


/* =========================================================
   ANALYTICS
========================================================= */

async function loadAnalytics() {
    try {
        const data = await getJSON("/api/analytics");

        updateAnalyticsStats(data);

        let historyData = { history: [] };

        try {
            historyData = await getJSON("/api/history");
        } catch {
            historyData = { history: [] };
        }

        renderAnalyticsCharts({
            ...data,
            sensor_history: historyData.history || []
        });

    } catch (error) {
        console.error("Analytics error:", error);
    }
}


function updateAnalyticsStats(data) {
    if (!data) return;

    const stats = data.stats || {};
    const distribution = data.fertility_distribution || [];

    function getCount(label) {
        const item = distribution.find(row =>
            String(row.fertility || "")
                .trim()
                .toLowerCase() === label.toLowerCase()
        );

        return item ? Number(item.count) || 0 : 0;
    }

    const total =
        data.total_predictions ??
        stats.total_predictions ??
        stats.total ??
        distribution.reduce(
            (sum, row) => sum + (Number(row.count) || 0),
            0
        );

    $("totalPredictions").textContent = total;

    $("highCount").textContent =
        stats.high ?? stats.high_count ?? getCount("High");

    $("mediumCount").textContent =
        stats.medium ?? stats.medium_count ?? getCount("Medium");

    $("lowCount").textContent =
        stats.low ?? stats.low_count ?? getCount("Low");
}


function renderAnalyticsCharts(data) {
    if (typeof Chart === "undefined") {
        console.error("Chart.js is not loaded.");
        return;
    }

    const fertilityCanvas = $("fertilityChart");

    if (fertilityCanvas) {
        if (fertilityChart) fertilityChart.destroy();

        const distribution = data.fertility_distribution || [];
        const labels = ["Low", "Medium", "High"];

        const values = labels.map(label => {
            const item = distribution.find(row =>
                String(row.fertility || "")
                    .trim()
                    .toLowerCase() === label.toLowerCase()
            );

            return item ? Number(item.count) || 0 : 0;
        });

        fertilityChart = new Chart(fertilityCanvas, {
            type: "bar",
            data: {
                labels,
                datasets: [{
                    label: "Predictions",
                    data: values,
                    borderWidth: 1,
                    borderRadius: 6,
                    maxBarThickness: 80
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: { precision: 0 }
                    }
                }
            }
        });
    }

    const sensorCanvas = $("sensorChart");

    if (sensorCanvas) {
        if (sensorChart) sensorChart.destroy();

        const history = Array.isArray(data.sensor_history)
            ? [...data.sensor_history].reverse()
            : [];

        const labels = history.map((row, index) =>
            row.timestamp
                ? formatTimestamp(row.timestamp)
                : `Reading ${index + 1}`
        );

        const moisture = history.map(row =>
            row.moisture == null ? null : Number(row.moisture)
        );

        const temperature = history.map(row =>
            row.temperature == null ? null : Number(row.temperature)
        );

        sensorChart = new Chart(sensorCanvas, {
            type: "line",
            data: {
                labels,
                datasets: [
                    {
                        label: "Moisture",
                        data: moisture,
                        borderWidth: 2,
                        tension: 0.25
                    },
                    {
                        label: "Temperature",
                        data: temperature,
                        borderWidth: 2,
                        tension: 0.25
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { beginAtZero: true }
                }
            }
        });
    }
}


/* =========================================================
   WEATHER
========================================================= */

async function loadWeather() {
    if (!currentLocation) return;

    try {
        const url =
            `/api/weather?latitude=${encodeURIComponent(
                currentLocation.latitude
            )}&longitude=${encodeURIComponent(
                currentLocation.longitude
            )}`;

        const data = await getJSON(url);

        const current =
            data.current ||
            data.weather?.current ||
            data.weather ||
            data;

        if (!current) return;

        const temp =
            current.temperature_2m ??
            current.temperature;

        const humidity =
            current.relative_humidity_2m ??
            current.humidity;

        $("weatherTemp").textContent =
            temp !== undefined
                ? `${formatNumber(temp, 1)}°`
                : "--°";

        $("weatherHumidity").textContent =
            humidity !== undefined
                ? `${formatNumber(humidity, 0)}%`
                : "--";

        $("weatherCondition").textContent =
            current.condition ||
            current.weather ||
            "Current conditions";

        $("weatherDetails").textContent =
            current.description ||
            "Local weather";

    } catch (error) {
        console.error("Weather error:", error);
    }
}


/* =========================================================
   CONNECTION
========================================================= */

async function checkBackend() {
    try {
        const response = await fetch("/api/health", {
            credentials: "same-origin"
        });

        if (!response.ok) throw new Error();

        updateConnection(true);

    } catch {
        updateConnection(false);
    }
}


function updateConnection(connected) {
    $("connectionText").textContent =
        connected ? "Backend connected" : "Backend unavailable";

    $("connectionSubtext").textContent =
        connected ? "System is ready" : "Check Flask server";

    const dot = document.querySelector(
        ".connection-card .status-dot"
    );

    if (dot) {
        dot.style.background = connected
            ? "var(--accent-strong)"
            : "var(--danger)";
    }
}


/* =========================================================
   THEME
========================================================= */

function setupTheme() {
    const saved = localStorage.getItem("soilai_theme");

    if (saved === "light") {
        document.body.classList.add("light-theme");
    }

    $("themeToggle").addEventListener("click", () => {
        document.body.classList.toggle("light-theme");

        localStorage.setItem(
            "soilai_theme",
            document.body.classList.contains("light-theme")
                ? "light"
                : "dark"
        );
    });
}


/* =========================================================
   REFRESH ALL
========================================================= */

async function refreshAll() {
    await checkBackend();
    await refreshLatest();

    updateLocationDisplay();

    await loadWeather();

    if (currentPage === "analytics") {
        await loadAnalytics();
    }

    if (currentPage === "history") {
        await loadHistory();
    }
}


/* =========================================================
   AUTO REFRESH
========================================================= */

setInterval(refreshLatest, 2000);

setInterval(checkBackend, 10000);

setInterval(loadWeather, 60000);

setInterval(() => {
    if (currentPage === "analytics") {
        loadAnalytics();
    }
}, 7000);

setInterval(() => {
    if (currentPage === "history") {
        loadHistory();
    }
}, 10000);


/* =========================================================
   INITIALIZATION
========================================================= */

document.addEventListener("DOMContentLoaded", () => {
    setupAuthTabs();

    $("loginForm").addEventListener("submit", login);
    $("signupForm").addEventListener("submit", signup);
    $("logoutBtn").addEventListener("click", logout);
    $("refreshHistoryBtn").addEventListener("click", loadHistory);

    const savedUser = localStorage.getItem("soilai_user");

    if (savedUser) {
        getJSON("/api/me")
            .then((data) => {
                if (data.authenticated && data.user) {
                    showApp(data.user);
                } else {
                    localStorage.removeItem("soilai_user");
                }
            })
            .catch(() => {
                localStorage.removeItem("soilai_user");
            });
    }
});