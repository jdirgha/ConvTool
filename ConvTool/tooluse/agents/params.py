"""Realistic parameter value generation for tool call arguments.

``generate_param_value`` dispatches through five layers (most-specific first):
endpoint-context override → exact name → keyword match → type fallback → string heuristics.
"""

from __future__ import annotations

import random
from typing import Any

_API_INTERNAL_PARAMS: frozenset[str] = frozenset({
    "function", "outputsize", "series_type", "series_type_1", "series_type_2",
    "format", "language", "lang", "locale", "fields", "type", "offset",
    "page", "page_size", "limit", "order", "sort", "referenceCurrencyUuid",
    "uuid", "key", "apikey", "api_key", "token", "access_token",
    "client_id", "client_secret", "app_id", "app_key",
})


def _is_api_internal(param_name: str) -> bool:
    """Return True for params that should never appear in user-facing messages.

    Covers the static blocklist plus Django ORM range suffixes (__lt, __gt, etc.).
    """
    if param_name in _API_INTERNAL_PARAMS:
        return True
    if any(param_name.endswith(sfx) for sfx in ("__lt", "__gt", "__lte", "__gte", "__in")):
        return True
    return False


_NAME_POOLS: dict[str, list] = {
    "location":     ["New York", "London", "Tokyo", "Paris", "Berlin", "Sydney"],
    "address":      ["123 Main St, New York", "10 Downing St, London", "1 Champs-Elysées, Paris"],
    "city":         ["San Francisco", "Munich", "Barcelona", "Toronto", "Singapore"],
    "origin":       ["Downtown Hotel", "Airport Terminal 1", "Central Station"],
    "destination":  ["Convention Center", "City Museum", "Botanical Garden"],
    "country":      ["United States", "Germany", "France", "Japan", "United Kingdom"],
    "country_code": ["US", "DE", "FR", "JP", "GB"],
    "state":        ["California", "New York", "Texas", "Florida", "Washington"],
    "zip":          ["10001", "90210", "60601", "30301", "94102"],
    "postcode":     ["SW1A 1AA", "W1A 0AX", "EC1A 1BB", "WC2N 5DU"],
    "postal_code":  ["10001", "75001", "10115", "E1 6RF"],
    "latitude":     [40.7128, 51.5074, 35.6762, 48.8566, 52.5200],
    "longitude":    [-74.0060, -0.1278, 139.6503, 2.3522, 13.4050],
    "ip":           ["8.8.8.8", "1.1.1.1", "208.67.222.222", "9.9.9.9"],
    "ip_address":   ["8.8.8.8", "1.1.1.1", "208.67.222.222", "9.9.9.9"],
    "icao":         ["KJFK", "EGLL", "LFPG", "RJTT", "KLAX", "EDDB", "OMDB", "ZBAA"],
    "iata":         ["JFK", "LHR", "CDG", "NRT", "LAX", "TXL", "DXB", "PEK"],
    "airport_code": ["JFK", "LHR", "CDG", "NRT", "LAX", "SFO", "ORD", "DXB"],
    "restaurant":   ["Trattoria Roma", "Le Petit Bistro", "Sakura Garden",
                     "Casa Del Sol", "The Golden Spoon", "Blue Nile"],
    "date":         ["2025-03-20", "2025-04-12", "2025-05-15", "2025-06-01"],
    "start_date":   ["2025-03-01", "2025-04-01", "2025-05-01"],
    "end_date":     ["2025-03-31", "2025-04-30", "2025-05-31"],
    "from_date":    ["2025-03-01", "2025-04-01"],
    "to_date":      ["2025-03-31", "2025-04-30"],
    "time":         ["09:00", "12:30", "14:00", "18:00", "19:30"],
    "check_in":     ["2025-03-15", "2025-04-01", "2025-05-20", "2025-06-10"],
    "check_out":    ["2025-03-18", "2025-04-05", "2025-05-25", "2025-06-14"],
    "symbol":       ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "BTC", "ETH", "META"],
    "ticker":       ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"],
    "stock_symbol": ["AAPL", "MSFT", "GOOGL", "TSLA"],
    "interval":     ["1day", "1week", "1h", "4h", "15min", "1month"],
    "time_period":  [14, 20, 50, 200],
    "outputsize":   [30, 50, 100],
    "series_type":  ["close", "open", "high", "low"],
    "series_type_1":["open", "close"],
    "series_type_2":["close", "high"],
    "format":       ["json", "csv"],
    "market":       ["USD", "EUR", "USDT", "BTC"],
    "function":     ["TIME_SERIES_DAILY", "TIME_SERIES_WEEKLY",
                     "DIGITAL_CURRENCY_DAILY", "CURRENCY_EXCHANGE_RATE"],
    "currency":     ["USD", "EUR", "GBP", "JPY", "CHF"],
    "currency_code":["USD", "EUR", "GBP"],
    "from_currency":["USD", "EUR", "GBP"],
    "to_currency":  ["EUR", "JPY", "CHF", "USD"],
    "base_currency":["USD", "EUR", "GBP"],
    "base":         ["USD", "EUR"],
    "target":       ["EUR", "JPY", "GBP"],
    "amount":       [100, 250, 500, 1000],
    "uuid":         ["Qwsogvtv82FCd", "razxDUgYGNAdQ", "HIVsRcGKkPFtW",
                     "aKzUVe4HhdWS", "ZlZpzOJo43mIo"],
    "referenceCurrencyUuid": ["yhjMzLPhuIDl"],
    "from_symbol":  ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD"],
    "to_symbol":    ["EUR", "JPY", "CHF", "GBP", "USD", "CAD", "AUD"],
    "hotel_id":     ["HTL-1042", "HTL-2891", "HTL-3374"],
    "room_type":    ["standard", "deluxe", "suite"],
    "guests":       [1, 2, 3, 4],
    "adults":       [1, 2],
    "children":     [0, 1, 2],
    "price_range":  ["budget", "moderate", "upscale"],
    "rating":       [3, 4, 5],
    "sort_by":      ["price", "rating", "distance", "relevance"],
    "origin_airport":      ["JFK", "LHR", "CDG", "NRT", "LAX"],
    "destination_airport": ["LHR", "CDG", "NRT", "SFO", "ORD"],
    "airline":             ["Delta", "Lufthansa", "British Airways", "ANA"],
    "flight_number":       ["DL123", "LH456", "BA789", "NH101"],
    "cabin_class":         ["economy", "business", "first"],
    "ingredient":   ["chicken", "pasta", "tomatoes", "garlic", "spinach"],
    "meal_type":    ["breakfast", "lunch", "dinner", "snack"],
    "cuisine":      ["italian", "japanese", "mexican", "indian", "french"],
    "diet":         ["vegetarian", "vegan", "keto", "paleo", "gluten-free"],
    "calories":     [400, 600, 800, 1200],
    "guest_name":   ["Alice Johnson", "Bob Smith", "Clara Martinez", "David Lee"],
    "name":         ["Alice", "Bob", "Clara", "David"],
    "name_1":       ["Emma", "James", "Sophia", "Liam"],
    "name_2":       ["Oliver", "Ava", "Noah", "Isabella"],
    "email":        ["user@example.com", "traveller@email.com", "guest@hotel.com"],
    "artist":       ["The Beatles", "Taylor Swift", "Daft Punk", "Coldplay"],
    "album":        ["Abbey Road", "Thriller", "Random Access Memories"],
    "genre":        ["pop", "rock", "jazz", "classical", "hip-hop", "electronic"],
    "platform":     ["pc", "mac", "ios", "android"],
    "game":         ["Chess", "Sudoku", "Minecraft", "FIFA"],
    "q":            ["Paris", "London", "New York", "Tokyo", "Berlin",
                     "Sydney", "Barcelona", "Singapore", "Toronto", "Dubai"],
    "query":        ["travel budget tips", "crypto market update", "hotel deals",
                     "restaurant near me", "outdoor activities"],
    "search":       ["flights to Paris", "hotels in Tokyo", "Italian restaurants",
                     "coffee shops downtown", "hiking trails nearby"],
    "keyword":      ["finance", "travel", "weather", "food", "entertainment"],
    "term":         ["budget travel", "currency exchange", "outdoor activities"],
    "limit":        [10, 20, 50],
    "page":         [1, 2, 3],
    "page_size":    [10, 20, 50],
    "offset":       [0, 10, 20],
    "order":        ["asc", "desc"],
    "sort":         ["price", "rating", "date", "relevance"],
    "language":     ["en", "de", "fr", "es", "ja"],
    "lang":         ["en", "de", "fr", "es"],
    "locale":       ["en-US", "de-DE", "fr-FR", "ja-JP"],
    "type":         ["json", "basic", "premium", "standard"],
    "fields":       ["id,name,description", "id,title,date", "name,price,rating"],
    "unitGroup":    ["metric", "us", "uk", "base"],
    "unit_group":   ["metric", "us", "uk"],
    "timesteps":    ["1h", "1d", "current"],
    "domain":       ["www.booking.com", "www.hotels.com", "www.airbnb.com"],
    "adults_number_by_rooms": ["1,1", "2,1", "2,2"],
    "text":         ["pasta with tomato", "grilled chicken", "chocolate cake",
                     "salmon sushi", "vegetarian lasagna"],
    "ingr":         ["chicken", "pasta", "tomato", "garlic", "salmon"],
    "start":        ["2025-03-01", "2025-04-01", "2025-05-01"],
    "end":          ["2025-03-31", "2025-04-30", "2025-05-31"],
    "from":         ["2025-03-01", "2025-04-01"],
    "to":           ["2025-03-31", "2025-04-30"],
    "dt":           ["2025-03-20", "2025-04-15", "2025-05-10"],
    "flnr":         ["DL123", "LH456", "BA789", "AA101"],
    "station":      ["JFK", "LHR", "CDG", "GVA", "ZRH"],
    "slug":         ["new-york", "london", "paris", "tokyo", "berlin"],
    "pickUpTime":   ["08:00", "10:00", "12:00", "14:00"],
    "dropOffTime":  ["18:00", "20:00", "22:00"],
    "pick_up_time": ["08:00", "10:00", "12:00"],
    "drop_off_time":["18:00", "20:00", "22:00"],
    "trend_type":   ["rising", "falling", "stable", "seasonal"],
    "m":            [1, 3, 6, 12],
    "mode":         ["driving", "walking", "transit", "cycling"],
    "units":        ["metric", "imperial"],
    "radius":       [1, 5, 10, 25],
    "title":        ["Team Standup", "Project Review", "Client Meeting", "Workshop"],
    "category":     ["restaurant", "vegetarian", "keto", "cafe", "vegan",
                     "hotel", "dessert", "main course", "park", "appetizer"],
    "food_category": ["vegetarian", "keto", "vegan", "dessert", "appetizer",
                      "main course", "salad", "soup", "breakfast"],
    "meal_category": ["vegetarian", "keto", "vegan", "dessert", "appetizer"],
    "duration":     [30, 60, 90, 120],
}

_KEYWORD_MATCHERS: list[tuple[str, str]] = [
    ("ip_address",       "ip_address"),
    ("from_symbol",      "from_symbol"),
    ("to_symbol",        "to_symbol"),
    ("unitgroup",        "unitGroup"),
    ("unit_group",       "unitGroup"),
    ("timesteps",        "timesteps"),
    ("trend_type",       "trend_type"),
    ("adults_number",    "adults_number_by_rooms"),
    ("pickup",           "pickUpTime"),
    ("dropoff",          "dropOffTime"),
    ("check_out",        "check_out"),
    ("checkout",         "check_out"),
    ("check_in",         "check_in"),
    ("checkin",          "check_in"),
    ("icao",             "icao"),
    ("iata",             "iata"),
    ("restaurant",       "restaurant"),
    ("station",          "station"),
    ("symbol",           "symbol"),
    ("interval",         "interval"),
    ("currency",         "currency"),
    ("country",          "country"),
    ("language",         "language"),
    ("format",           "format"),
    ("latitude",         "latitude"),
    ("longitude",        "longitude"),
    ("date",             "date"),
    ("city",             "city"),
    ("location",         "location"),
    ("airport",          "airport_code"),
    ("limit",            "limit"),
    ("page",             "page"),
    ("sort",             "sort"),
    ("genre",            "genre"),
    ("query",            "query"),
    ("market",           "market"),
    ("ingredient",       "ingredient"),
    ("ingr",             "ingr"),
    ("cuisine",          "cuisine"),
    ("artist",           "artist"),
    ("domain",           "domain"),
    ("slug",             "slug"),
    ("trend",            "trend_type"),
]

_ENDPOINT_PARAM_POOLS: dict[tuple[str, str], list] = {
    ("address", "text"):     ["123 Main Street, London", "10 Downing Street, London",
                              "Champs-Élysées 1, Paris", "5th Avenue, New York",
                              "Potsdamer Platz 1, Berlin"],
    ("address", "query"):    ["London", "Paris", "New York", "Tokyo", "Berlin",
                              "Barcelona", "Sydney", "Toronto"],
    ("address", "q"):        ["London", "Paris", "New York", "Tokyo", "Berlin"],
    ("food", "text"):        ["pasta carbonara", "grilled salmon", "chicken tikka masala",
                              "chocolate mousse", "caesar salad", "beef stir-fry"],
    ("food", "query"):       ["quick pasta recipe", "low-carb dinner", "vegan dessert",
                              "keto breakfast", "easy cocktail recipe"],
    ("food", "q"):           ["pasta", "salmon", "chicken", "chocolate cake", "salad"],
    ("food", "ingr"):        ["chicken", "pasta", "salmon", "spinach", "avocado"],
    ("food", "ingredient"):  ["chicken", "pasta", "tomato", "garlic", "salmon"],
    ("weather", "q"):        ["London", "Paris", "New York", "Tokyo", "Berlin", "Sydney"],
    ("weather", "query"):    ["weather in London", "forecast Paris", "current conditions Tokyo"],
    ("weather", "location"): ["London, UK", "Paris, France", "New York, USA", "Tokyo, Japan"],
    ("weather", "text"):     ["London", "Paris", "New York"],
    ("flight", "q"):         ["JFK", "LHR", "CDG", "NRT", "LAX"],
    ("flight", "query"):     ["New York to London", "Paris to Tokyo", "JFK LHR"],
    ("flight", "text"):      ["New York JFK", "London Heathrow", "Paris CDG"],
    ("finance", "q"):        ["AAPL", "GOOGL", "BTC", "ETH", "EUR/USD"],
    ("finance", "query"):    ["Apple stock price", "Bitcoin price", "EUR USD rate"],
    ("hotel", "query"):      ["hotels in London", "Paris accommodation", "Tokyo hotel deals"],
    ("hotel", "q"):          ["London", "Paris", "Tokyo", "New York", "Berlin"],
    ("location", "query"):   ["London", "Paris", "New York", "51.5074,-0.1278"],
    ("location", "q"):       ["London", "Paris", "New York", "Tokyo"],
    ("location", "text"):    ["London, UK", "Paris, France", "New York, USA"],
}


def _get_endpoint_context(endpoint_id: str) -> str:
    """Map an endpoint ID to a broad context label (food, weather, finance, etc.)."""
    ep = endpoint_id.lower()
    if any(k in ep for k in ("geocod", "address completion", "reverse geocod",
                              "postcode", "postal", "zip", "uk postcode")):
        return "address"
    if any(k in ep for k in ("recipe", "keto", "cocktail", "pizza", "food",
                              "nutrition", "edamam", "tasty", "bbc good", "ingredient")):
        return "food"
    if any(k in ep for k in ("weather", "forecast", "climate", "meteostat",
                              "visual crossing", "tomorrow.io", "open weather",
                              "national weather", "air quality")):
        return "weather"
    if any(k in ep for k in ("flight", "airport", "airline", "flightera",
                              "priceline", "kayak", "airportstat")):
        return "flight"
    if any(k in ep for k in ("stock", "exchange rate", "currency", "crypto",
                              "coinranking", "alpha vantage", "twelve data",
                              "real-time finance", "stock analysis", "wyre")):
        return "finance"
    if any(k in ep for k in ("hotel", "booking", "tripadvisor", "hotels.com",
                              "hotels com", "accommodation")):
        return "hotel"
    if any(k in ep for k in ("movie", "imdb", "netflix", "steam", "epic games",
                              "deezer", "music", "love calculator", "chuck norris")):
        return "entertainment"
    if any(k in ep for k in ("timezone", "geolocation", "ip geo", "country",
                              "geocode", "reverse geocoding", "rest country")):
        return "location"
    return ""


def _param_from_endpoint_context(
    name: str, endpoint_id: str, rng: random.Random
) -> Any | None:
    ctx = _get_endpoint_context(endpoint_id)
    if not ctx:
        return None
    pool = _ENDPOINT_PARAM_POOLS.get((ctx, name))
    return rng.choice(pool) if pool else None


def _param_from_name(name: str, rng: random.Random) -> Any | None:
    pool = _NAME_POOLS.get(name)
    return rng.choice(pool) if pool is not None else None


def _param_from_keyword(name: str, rng: random.Random) -> Any | None:
    name_lower = name.lower()
    for keyword, pool_key in _KEYWORD_MATCHERS:
        if keyword in name_lower:
            pool = _NAME_POOLS.get(pool_key)
            if pool is not None:
                return rng.choice(pool)
    return None


def _param_from_type(name: str, ptype: str, rng: random.Random) -> Any | None:
    name_lower = name.lower()
    if ptype == "NUMBER":
        if any(k in name_lower for k in ("id", "uuid")):
            return rng.randint(100, 99999)
        if "amount" in name_lower:
            return rng.choice([100, 250, 500, 1000])
        if any(k in name_lower for k in ("size", "guest", "adult", "child")):
            return rng.randint(1, 6)
        if any(k in name_lower for k in ("day", "limit", "count")):
            return rng.randint(1, 7)
        if "radius" in name_lower:
            return rng.choice([1, 5, 10])
        if "duration" in name_lower:
            return rng.choice([30, 60, 90, 120])
        return rng.randint(1, 100)
    if ptype == "BOOLEAN":
        return rng.choice([True, False])
    if ptype == "ARRAY":
        return [f"item_{rng.randint(1, 100)}"]
    if ptype == "OBJECT":
        return {"key": f"value_{rng.randint(1, 100)}"}
    return None


def _param_string_heuristics(name: str, rng: random.Random) -> Any:
    """Last-resort string value using common name-pattern heuristics."""
    name_lower = name.lower()
    if any(k in name_lower for k in ("id", "uuid", "ref", "code")):
        return f"{name_lower[:4].upper()}-{rng.randint(1000, 9999)}"
    if "name" in name_lower:
        return rng.choice(["Alice", "Bob", "Clara", "David"])
    if "email" in name_lower:
        return "user@example.com"
    if "url" in name_lower or "link" in name_lower:
        return "https://example.com/resource"
    if "key" in name_lower or "token" in name_lower:
        return f"key_{rng.randint(1000, 9999)}"
    # Short / abbreviated param names
    if name_lower == "dt":
        return rng.choice(["2025-03-20", "2025-04-15", "2025-05-10"])
    if name_lower == "flnr":
        return rng.choice(["DL123", "LH456", "BA789"])
    if name_lower == "m":
        return rng.choice([1, 3, 6, 12])
    if name_lower in ("start", "from"):
        return rng.choice(["2025-03-01", "2025-04-01", "2025-05-01"])
    if name_lower in ("end", "to"):
        return rng.choice(["2025-03-31", "2025-04-30", "2025-05-31"])
    if name_lower == "text":
        return rng.choice(["grilled chicken", "pasta primavera", "chocolate cake"])
    if name_lower == "ingr":
        return rng.choice(["chicken", "pasta", "tomato", "garlic"])
    return f"value_{name_lower}"


def generate_param_value(
    name: str,
    ptype: str,
    rng: random.Random,
    endpoint_id: str = "",
) -> Any:
    """Generate a realistic argument value by name, type, and endpoint context."""
    if endpoint_id:
        val = _param_from_endpoint_context(name, endpoint_id, rng)
        if val is not None:
            return val

    val = _param_from_name(name, rng)
    if val is not None:
        return val

    val = _param_from_keyword(name, rng)
    if val is not None:
        return val

    val = _param_from_type(name, ptype, rng)
    if val is not None:
        return val

    return _param_string_heuristics(name, rng)
