"""
Weather Plugin for Sentra.

Provides real-time weather data lookup via the Open-Meteo API (free, no key required).
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

PLUGIN_NAME = "weather-plugin"
PLUGIN_VERSION = "1.0.0"


async def get_weather(location: str) -> dict[str, Any]:
    """
    Fetch current weather for a location using Open-Meteo's geocoding + weather API.

    Args:
        location: City name or place (e.g., "London", "Tokyo").

    Returns:
        Dict with temperature, conditions, wind, humidity, etc.
    """
    import json
    import urllib.request

    try:
        # Step 1: Geocode location
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1"
        with urllib.request.urlopen(geo_url, timeout=5) as resp:
            geo_data = json.loads(resp.read().decode())

        results = geo_data.get("results", [])
        if not results:
            return {"error": f"Location '{location}' not found"}

        lat = results[0]["latitude"]
        lon = results[0]["longitude"]
        name = results[0].get("name", location)
        country = results[0].get("country", "")

        # Step 2: Fetch weather
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code"
            f"&temperature_unit=celsius"
        )
        with urllib.request.urlopen(weather_url, timeout=5) as resp:
            weather_data = json.loads(resp.read().decode())

        current = weather_data.get("current", {})
        return {
            "location": f"{name}, {country}",
            "temperature_celsius": current.get("temperature_2m"),
            "humidity_percent": current.get("relative_humidity_2m"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
            "weather_code": current.get("weather_code"),
            "source": "Open-Meteo",
        }

    except Exception as e:
        logger.error("Weather fetch failed: %s", e)
        return {"error": str(e)}


def setup(agent: Any) -> None:
    """Called when the plugin is loaded by SentraAgent."""
    logger.info("Weather plugin loaded for agent '%s'", getattr(agent, "name", "unknown"))
