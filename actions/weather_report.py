"""
Weather Report action — JARVIS 6.3.

Uses the OpenWeatherMap API for current conditions when an API key is
configured.  Falls back to the previous behaviour (Google search via
browser) when no key is available so the feature always remains usable.

If no city is supplied in the ``parameters``, the action looks up the user's
stored home city from memory (``location.home_city`` or ``identity.city``).
"""

import json
from urllib.parse import quote_plus

try:
    import requests
except ImportError:
    requests = None

from core.retry import retry
from memory.config_manager import load_api_keys
from memory.memory_manager import load_memory


# OpenWeatherMap API endpoint
_OWM_BASE = "https://api.openweathermap.org/data/2.5/weather"
_TIMEOUT = 8


def _owm_key() -> str | None:
    """Return the stored OpenWeatherMap API key, or None."""
    try:
        return load_api_keys().get("weather_api_key")
    except Exception:
        return None


def _get_home_city(user_name: str = "") -> str | None:
    """Look up the user's remembered home city from memory."""
    try:
        mem = load_memory()
        # Check identity.city, then location.home_city
        ident = mem.get("identity", {}) or {}
        city = ident.get("city")
        if city and isinstance(city, str):
            return city.strip()
        loc = mem.get("location", {}) or {}
        city = loc.get("home_city")
        if city and isinstance(city, str):
            return city.strip()
    except Exception:
        pass
    return None


def _format_forecast(data: dict) -> str:
    """Turn a raw OpenWeatherMap JSON response into spoken text."""
    try:
        name = data.get("name", "your location")
        sys_info = data.get("sys", {})
        country = sys_info.get("country", "")
        city_str = f"{name}, {country}" if country else name

        weather = data.get("weather", [])
        if weather:
            desc = weather[0].get("description", "conditions unknown")
        else:
            desc = "conditions unknown"

        main = data.get("main", {})
        temp_k = main.get("temp")
        feels_k = main.get("feels_like")
        humidity = main.get("humidity")
        wind = data.get("wind", {})
        wind_speed = wind.get("speed")

        temp_c = temp_k - 273.15 if temp_k else 0
        feels_c = feels_k - 273.15 if feels_k else temp_c

        parts = [
            f"Weather for {city_str}, sir.",
            f"Currently {desc}.",
            f"Temperature is {temp_c:.0f} degrees Celsius",
        ]
        if feels_k is not None:
            parts.append(f"feels like {feels_c:.0f}")
        if humidity is not None:
            parts.append(f"humidity {humidity}%")
        if wind_speed is not None:
            parts.append(f"wind {wind_speed} meters per second")
        parts.append("Stay dry and have a great day.")
        return ". ".join(parts) + "."
    except Exception:
        return f"Weather data received for your location, sir."


@retry(on_exceptions=(Exception,), tries=3, delay=1.0, backoff=2.0)
def _fetch_from_api(city: str) -> str | None:
    """Query OpenWeatherMap for *city*.  Returns spoken text or None on failure."""
    if requests is None:
        return None
    key = _owm_key()
    if not key:
        return None

    try:
        url = f"{_OWM_BASE}?q={quote_plus(city)}&appid={key}&units=metric"
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if data.get("cod") != 200 and data.get("cod") != "200":
            return None
        return _format_forecast(data)
    except Exception as exc:
        from core import logger as _logger
        try:
            _logger.debug("weather API error: %s", exc)
        except Exception:
            pass
        return None


def weather_action(
    parameters: dict,
    player=None,
    session_memory=None,
    user_name: str = "",
) -> str:
    city = parameters.get("city") or parameters.get("location")
    when = parameters.get("time", "today")

    # JARVIS 6.3 — if no city given, look in memory for the user's home city
    if not city or not isinstance(city, str) or not city.strip():
        city = _get_home_city(user_name)

    if not city or not city.strip():
        msg = "Sir, I don't know which city to check. Please tell me the city."
        _log(msg, player)
        return msg

    city = city.strip()
    when = (when or "today").strip()

    # Attempt API-based weather first (JARVIS 6.3)
    api_result = _fetch_from_api(city)
    if api_result:
        _log(api_result, player)
        if session_memory:
            try:
                session_memory.set_last_search(query=f"weather {city} {when}", response=api_result)
            except Exception:
                pass
        return api_result

    # No API key or API failed — return an error instead of opening browser
    msg = f"Sir, I couldn't fetch live weather for {city}. Please ensure a weather API key is configured."
    _log(msg, player)
    if session_memory:
        try:
            session_memory.set_last_search(query=f"weather {city} {when}", response=msg)
        except Exception:
            pass
    return msg


def _log(message: str, player=None) -> None:
    print(f"[Weather] {message}")
    if player:
        try:
            player.write_log(f"JARVIS: {message}")
        except Exception:
            pass
