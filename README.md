# ✈️ SkyTrek — Interactive Flight Planner

An interactive flight planning and booking simulator built in Python with Streamlit,
mapping **57,421 airports worldwide** and turning raw aviation data into a working
end-to-end travel booking experience.

Explore airports on a 3D globe, plan a route between any two cities with
distance-based pricing, pull live flight offers from Google Flights, add fares to a
cart, and check out with a downloadable boarding pass.

---

## Features

### 🌍 Airport Explorer
Filter 57,000+ airports by country, facility type, and elevation, then render the
results on an interactive PyDeck map. Includes an auto-locate option that finds the
nearest major airports to your current position via IP geolocation.

### 🗺️ Route Planner
Pick an origin and destination and the app computes true great-circle distance using
the **Haversine formula**, then prices the fare from that distance, estimates flight
time, and draws the route as a 3D arc across the globe.

### ⏱️ Real-Time Flight Search
Queries the **SerpApi Google Flights API** for live fares on a chosen date, and the
**FlightAware AeroAPI** for real gate and terminal assignments. Both are optional —
without API keys the app falls back to realistic simulated results, so it always runs.

### 🛒 Cart & Checkout
A stateful shopping cart backed by Streamlit's session state. Add multiple flights,
adjust passenger details, remove items, and generate a formatted **downloadable
boarding pass** at checkout.

### 📊 Data Insights
Four analytical views of the global dataset:
1. **Global airport density heatmap** — PyDeck HeatmapLayer over all coordinates
2. **Facility type breakdown** — pie chart of active airports by type
3. **Airport types by country** — horizontal bar comparison
4. **Regional breakdown** — pandas pivot table by continent and facility type

---

## Tech Stack

| Layer | Tools |
|---|---|
| App framework | Streamlit |
| Data | pandas, NumPy |
| Mapping | PyDeck (Deck.gl) — Scatterplot, Arc, and Heatmap layers |
| Charts | Matplotlib |
| Live APIs | SerpApi (Google Flights), FlightAware AeroAPI, IP-API |

---

## Running It Locally

```bash
git clone https://github.com/ayertime/skytrek-flight-planner.git
cd skytrek-flight-planner

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. The airport dataset is included in
`data/`, so it works immediately with no additional setup.

### Optional — enabling live flight data

The app runs fully without API keys. To turn on live flight search and real gate
lookups instead of simulated data:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Then add your keys to that file:

```toml
SERPAPI_API_KEY = "your-key-here"
FLIGHTAWARE_API_KEY = "your-key-here"
```

Free tiers: [SerpApi](https://serpapi.com/google-flights-api) ·
[FlightAware AeroAPI](https://flightaware.com/commercial/aeroapi/)

> `secrets.toml` is gitignored and is never committed.

---

## Data

[**OurAirports / OpenFlights**](https://openflights.org/data.html) — 57,421 airports,
heliports, and airfields with ICAO/IATA identifiers, coordinates, elevation, facility
type, and region. Included in this repo as `data/airports.csv` (~6 MB).

---

## Project Origin

Originally built as the final project for **CS 230 at Bentley University**, where the
brief was to build an interactive, data-driven Streamlit application that tells a story
with a real-world dataset. This repository is a cleaned-up release of that project:
credentials moved out of source, data paths made portable, and the code and
documentation prepared for public use.

## AI Assistance

Portions of this project's layout, styling, and feature scaffolding were developed with
the assistance of AI tools — **Claude** and **Gemini**. All code was reviewed, modified,
and integrated by the author.

## License

MIT — see [LICENSE](LICENSE).
