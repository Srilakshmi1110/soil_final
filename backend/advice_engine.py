"""
SoilAI advice engine.

Provides rule-based, explainable soil and crop guidance from measured/model
inputs. This module does not generate or alter sensor readings or ML predictions.

Expected inputs are a dictionary containing any available values among:
ph, moisture, nitrogen, cec, temperature, fertility.
"""

from typing import Any, Dict, List, Optional


class SoilAdviceEngine:
    """Generate practical soil-improvement and crop-screening advice."""

    CROP_PROFILES = {
        "rice": {
            "ph": (5.5, 7.0),
            "moisture": (60, 100),
            "nitrogen": (80, 200),
            "note": "Rice generally needs consistently moist soil and adequate nitrogen."
        },
        "wheat": {
            "ph": (6.0, 7.5),
            "moisture": (30, 60),
            "nitrogen": (80, 180),
            "note": "Wheat generally performs well in near-neutral soil with moderate moisture."
        },
        "maize": {
            "ph": (5.8, 7.0),
            "moisture": (40, 70),
            "nitrogen": (100, 200),
            "note": "Maize has a relatively high nitrogen requirement and needs well-drained soil."
        },
        "groundnut": {
            "ph": (5.5, 7.0),
            "moisture": (30, 60),
            "nitrogen": (20, 100),
            "note": "Groundnut generally prefers well-drained, slightly acidic to neutral soil."
        },
        "millet": {
            "ph": (5.5, 7.5),
            "moisture": (15, 50),
            "nitrogen": (30, 120),
            "note": "Millets are comparatively tolerant of lower soil moisture."
        },
        "cotton": {
            "ph": (5.8, 8.0),
            "moisture": (35, 65),
            "nitrogen": (60, 160),
            "note": "Cotton prefers well-drained soil and moderate moisture."
        },
        "sugarcane": {
            "ph": (6.0, 7.5),
            "moisture": (50, 85),
            "nitrogen": (100, 220),
            "note": "Sugarcane has a long growing period and needs sustained moisture and nutrients."
        }
    }

    def __init__(self):
        pass

    @staticmethod
    def _number(data: Dict[str, Any], key: str) -> Optional[float]:
        value = data.get(key)
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _in_range(value: Optional[float], bounds) -> Optional[bool]:
        if value is None:
            return None
        return bounds[0] <= value <= bounds[1]

    def generate_advice(self, soil_data: Dict[str, Any]) -> Dict[str, Any]:
        """Return soil-specific advice, limitations, and crop screening."""
        data = soil_data or {}
        ph = self._number(data, "ph")
        moisture = self._number(data, "moisture")
        nitrogen = self._number(data, "nitrogen")
        cec = self._number(data, "cec")
        temperature = self._number(data, "temperature")

        advice: List[str] = []
        warnings: List[str] = []

        if ph is None:
            warnings.append("pH data is unavailable; pH-related advice was skipped.")
        elif ph < 5.5:
            advice.append(
                "Soil is strongly acidic. Consider a soil-test-based lime "
                "recommendation and retest pH before application."
            )
        elif ph < 6.0:
            advice.append(
                "Soil is mildly acidic. Check crop-specific pH requirements; "
                "lime may be useful only if recommended by a soil test."
            )
        elif ph <= 7.5:
            advice.append("The measured pH is within a range suitable for many crops.")
        elif ph <= 8.0:
            advice.append(
                "Soil is mildly alkaline. Check nutrient availability, especially "
                "iron, zinc and phosphorus, and confirm with a soil test."
            )
        else:
            advice.append(
                "Soil is alkaline. Test salinity and micronutrients before choosing "
                "any amendment; avoid applying gypsum unless sodicity is confirmed."
            )

        if moisture is None:
            warnings.append("Moisture data is unavailable; irrigation advice was skipped.")
        elif moisture < 20:
            advice.append(
                "Measured moisture is low. Consider irrigation based on crop stage "
                "and local soil conditions; avoid waterlogging."
            )
        elif moisture < 35:
            advice.append(
                "Moisture is on the lower side. Monitor the root zone and irrigate "
                "when appropriate for the crop."
            )
        elif moisture <= 70:
            advice.append("Measured moisture is moderate for many field-crop situations.")
        else:
            advice.append(
                "Measured moisture is high. Check drainage and avoid excess irrigation."
            )

        if nitrogen is None:
            warnings.append("Nitrogen data is unavailable; nitrogen advice was skipped.")
        elif nitrogen < 50:
            advice.append(
                "Raster nitrogen value is relatively low. Confirm with a representative "
                "soil test before planning split nitrogen applications."
            )
        elif nitrogen < 100:
            advice.append(
                "Nitrogen appears moderate. Match fertilizer timing and quantity to "
                "the crop recommendation and a laboratory soil test."
            )
        else:
            advice.append(
                "Nitrogen raster value is not low under this screening threshold. "
                "Avoid adding nitrogen without checking crop demand and soil-test results."
            )

        if cec is None:
            warnings.append("CEC data is unavailable; nutrient-retention advice was skipped.")
        elif cec < 10:
            advice.append(
                "CEC is relatively low, suggesting limited nutrient-retention capacity. "
                "Consider organic matter practices and split nutrient applications."
            )
        elif cec <= 25:
            advice.append(
                "CEC is in a moderate screening range. Maintain organic matter and "
                "base fertilizer decisions on soil testing."
            )
        else:
            advice.append(
                "CEC is relatively high, indicating greater potential nutrient retention. "
                "Continue balanced nutrient management."
            )

        if temperature is not None:
            if temperature < 10:
                advice.append("Soil temperature is low; germination and nutrient uptake may be slower.")
            elif temperature > 35:
                advice.append("Soil temperature is high; monitor crop heat stress and soil moisture.")

        crops = self.recommend_crops(data)

        return {
            "advice": advice,
            "warnings": warnings,
            "crop_recommendations": crops,
            "input_values": {
                "ph": ph,
                "moisture": moisture,
                "nitrogen": nitrogen,
                "cec": cec,
                "temperature": temperature,
                "fertility": data.get("fertility")
            },
            "disclaimer": (
                "Rule-based screening only. Crop suitability and fertilizer rates "
                "must be confirmed using local agronomic guidance and a laboratory soil test."
            )
        }

    def recommend_crops(self, soil_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank crops by how many available measurements fall in profile ranges."""
        data = soil_data or {}
        values = {
            "ph": self._number(data, "ph"),
            "moisture": self._number(data, "moisture"),
            "nitrogen": self._number(data, "nitrogen")
        }

        ranked = []
        for crop, profile in self.CROP_PROFILES.items():
            checks = []
            for key in ("ph", "moisture", "nitrogen"):
                actual = values[key]
                if actual is not None:
                    checks.append(self._in_range(actual, profile[key]))

            if not checks:
                continue

            matched = sum(checks)
            score = round(100 * matched / len(checks), 1)
            ranked.append({
                "crop": crop.title(),
                "score": score,
                "matched_parameters": matched,
                "evaluated_parameters": len(checks),
                "reason": profile["note"]
            })

        ranked.sort(key=lambda item: (item["score"], item["matched_parameters"]), reverse=True)
        return ranked[:5]

    # Compatibility aliases for projects that call different method names.
    def get_advice(self, soil_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.generate_advice(soil_data)

    def get_recommendations(self, soil_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        return self.recommend_crops(soil_data)


def generate_advice(soil_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for simple imports."""
    return SoilAdviceEngine().generate_advice(soil_data)
