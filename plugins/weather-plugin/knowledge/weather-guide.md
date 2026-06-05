# Weather Plugin — User Guide

## Overview

The Weather Plugin provides real-time weather data for Sentra using the **Open-Meteo API** — completely free, no API key required. All weather lookups are geocoded automatically from location names.

---

## How It Works

```
Location Name → Geocoding API → Coordinates → Weather API → Formatted Data
```

1. You provide a location name (e.g., "London", "Tokyo", "New York")
2. Open-Meteo's geocoding endpoint resolves it to latitude/longitude
3. The weather API returns current conditions and forecast data
4. Results are formatted for easy consumption by the AI

---

## Data Available

### Current Conditions

| Field | Unit | Description |
|---|---|---|
| temperature | °C | Current temperature |
| feels_like | °C | Apparent temperature (wind chill/heat index) |
| humidity | % | Relative humidity |
| wind_speed | km/h | Wind speed at 10m height |
| wind_direction | ° | Wind direction in degrees |
| precipitation | mm | Current precipitation amount |
| weather_code | WMO code | Weather condition code |
| cloud_cover | % | Cloud coverage percentage |
| pressure | hPa | Surface pressure |
| uv_index | 0-11+ | UV radiation index |
| visibility | km | Visibility distance |

### Weather Codes (WMO)

| Code | Condition |
|---|---|
| 0 | Clear sky |
| 1-3 | Mainly clear / Partly cloudy / Overcast |
| 45, 48 | Fog / Depositing rime fog |
| 51-55 | Drizzle (light / moderate / dense) |
| 61-65 | Rain (slight / moderate / heavy) |
| 71-75 | Snow (slight / moderate / heavy) |
| 80-82 | Rain showers (slight / moderate / violent) |
| 95 | Thunderstorm |
| 96, 99 | Thunderstorm with hail |

---

## API Details

- **Provider**: [Open-Meteo](https://open-meteo.com/)
- **Pricing**: Free for non-commercial use (up to 10,000 requests/day)
- **No API key required**
- **Geocoding**: Automatic via Open-Meteo Geocoding API
- **Timeout**: 5 second timeout per request
- **Rate limit**: 600 requests/minute

---

## Location Formats

The plugin accepts flexible location inputs:
- **City name**: "London", "Paris", "Tokyo"
- **City + Country**: "London, UK" vs "London, Canada"
- **Full address**: "1600 Pennsylvania Avenue, Washington DC"

For ambiguous names, the geocoding API returns the most populous match. To get a specific location, include the country name.

---

## Limitations

- **Forecast**: Currently returns current conditions only (not multi-day forecast)
- **Historical**: No historical weather data access
- **Offline**: Requires internet connectivity
- **Accuracy**: Weather data is modeled, not from a local weather station
