"""
SkyTrek: Flight Planner
=======================

An interactive flight planning and booking simulator built with Streamlit.

Explore 57,000+ airports worldwide on an interactive map, plan routes between
cities with distance-based pricing, search live flight offers through Google
Flights, and simulate a booking end to end with a shopping cart and a
downloadable boarding pass. A Data Insights page adds a global airport
heatmap, facility-type breakdowns, and regional pivot tables.

Author:  PJ Civitarese
Origin:  Built as the final project for CS 230 (Bentley University), then
         cleaned up and packaged for release.
Data:    OurAirports / OpenFlights - Airports Around the World

Data sources and libraries:
- OurAirports / OpenFlights Airport Database - https://openflights.org/data.html
- Streamlit - https://docs.streamlit.io/
- PyDeck (Deck.gl for Python) - https://deckgl.readthedocs.io/en/latest/
- SerpApi Google Flights API - https://serpapi.com/google-flights-api
- FlightAware AeroAPI - https://flightaware.com/commercial/aeroapi/
- IP-API (geolocation) - http://ip-api.com/
- pandas, NumPy, Matplotlib

Development note:
Portions of this project's layout, styling, and feature scaffolding were
developed with the assistance of AI tools (Claude and Gemini). All code was
reviewed, modified, and integrated by the author. See README.md for details.
"""

import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import random
import time
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

import requests  # FlightAware API calls

# Resolve data relative to this file so the app runs from any directory.
DATA_PATH = Path(__file__).parent / "data" / "airports.csv"

try:
    from serpapi import GoogleSearch
except ImportError:
    GoogleSearch = None

# Page Configuration
st.set_page_config(
    page_title="SkyTrek: Flight Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# API KEYS
#
# Keys are read from Streamlit's secrets store, never hardcoded here.
# Copy .streamlit/secrets.toml.example to .streamlit/secrets.toml and
# fill in your own keys. That file is gitignored and never committed.
#
# Both keys are OPTIONAL - the app runs fully without them. Live flight
# search and live gate lookups simply fall back to simulated data.
# ------------------------------------------------------------------
def _secret(name: str, default: str = "") -> str:
    """Read a key from st.secrets, returning a default if it isn't set."""
    try:
        return st.secrets.get(name, default)
    except Exception:
        # No secrets.toml present - run in offline/simulated mode.
        return default


API_FLIGHTAWARE_KEY = _secret("FLIGHTAWARE_API_KEY")  # AeroAPI - live gate data
API_SERPAPI_KEY = _secret("SERPAPI_API_KEY")  # SerpApi - live Google Flights
# ------------------------------------------------------------------

# [ST3] Custom Styling (Sidebar and general look)
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)


# Data Loading
@st.cache_data
def load_data():
    """
    Loads and normalizes the airport dataset from CSV.

    Returns an empty DataFrame if the dataset is missing, so the app can
    surface a clear error instead of crashing.
    """
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        st.error(
            f"Could not find the airport dataset at `{DATA_PATH}`. "
            "Make sure `data/airports.csv` exists - see the README for setup."
        )
        return pd.DataFrame()

    # [COLUMNS] Clean up column names to handle different CSV formats
    # [LISTCOMP] List comprehension to clean column names
    df.columns = [c.strip().lower() for c in df.columns]

    # [FIX] Remove duplicate columns (keep first occurrence)
    df = df.loc[:, ~df.columns.duplicated()]

    # 1. Rename simple 1-to-1 mappings
    rename_map = {
        'latitude': 'lat',
        'longitude': 'lon',
        'airport': 'name',
        'airport name': 'name',
        'airport_name': 'name',
        'iso_country': 'country',
        'elevation_ft': 'altitude',
        'elevation': 'altitude',
        'alt': 'altitude'
    }
    df = df.rename(columns=rename_map)

    # 2. Handle Coordinates (Lat/Lon) parsing if missing
    # Note: User data format is "Lon, Lat" (e.g., "-74.9..., 40.0...")
    if 'lat' not in df.columns or 'lon' not in df.columns:
        if 'coordinates' in df.columns:
            try:
                # Remove quotes and split
                coords = df['coordinates'].astype(str).str.replace('"', '').str.split(',', expand=True)
                if coords.shape[1] >= 2:
                    df['lon'] = pd.to_numeric(coords[0].str.strip(), errors='coerce')  # Index 0 is Longitude
                    df['lat'] = pd.to_numeric(coords[1].str.strip(), errors='coerce')  # Index 1 is Latitude
                    st.success("Successfully parsed 'coordinates' column.")
            except Exception as e:
                st.error(f"Error parsing 'coordinates' column: {e}")

    # 3. Smart Fill for IATA (Coalesce)
    # Priority: iata_code > local_code > gps_code > ident
    if 'iata' not in df.columns:
        df['iata'] = np.nan

    for col in ['iata_code', 'local_code', 'gps_code', 'ident']:
        if col in df.columns:
            df['iata'] = df['iata'].fillna(df[col])

    # 4. Smart Fill for City (Coalesce)
    # Priority: municipality > iso_region
    if 'city' not in df.columns:
        df['city'] = np.nan

    for col in ['municipality', 'iso_region']:
        if col in df.columns:
            df['city'] = df['city'].fillna(df[col])

    # 5. Ensure required columns exist and have defaults
    required_cols = ['lat', 'lon', 'name', 'city', 'country', 'iata', 'altitude']
    for col in required_cols:
        if col not in df.columns:
            if col == 'altitude':
                df[col] = 0
            else:
                df[col] = 'Unknown'

    # 6. Extract State/Region from iso_region
    # Format is usually "US-CA", "GB-ENG", etc.
    if 'iso_region' in df.columns:
        df['state'] = df['iso_region'].astype(str).apply(lambda x: x.split('-')[1] if '-' in x else x)
    else:
        df['state'] = 'Unknown'

    # [FILTER1] Filter out any rows with missing lat/lon
    df = df.dropna(subset=['lat', 'lon'])

    # Ensure consistent data types (strings) for text columns
    text_cols = ['name', 'city', 'country', 'iata', 'state']
    for col in text_cols:
        df[col] = df[col].fillna('Unknown').astype(str)

    return df


# Helper Functions
def get_min_max_altitude(df):
    """
    [FUNCRETURN2] Returns two values: min and max altitude from the dataframe.
    """
    if not df.empty and 'altitude' in df.columns:
        return int(df['altitude'].min()), int(df['altitude'].max())
    return 0, 10000


def calculate_distance(lat1, lon1, lat2, lon2, radius=3958.8):
    """
    [FUNC2P] Calculates distance. 'radius' has a default value.
    Returns: Distance in miles (float)
    """
    R = radius

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2) ** 2 + \
        np.cos(phi1) * np.cos(phi2) * \
        np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


# [COMPLEXITY] Helper to get user location via API (No Key Required)
def get_user_location_by_ip():
    """
    Fetches user's approximate location using free IP-API.
    Returns: lat (float), lon (float)
    """
    try:
        # Note: http is required for the free endpoint (no SSL)
        response = requests.get("http://ip-api.com/json/")
        data = response.json()
        if data['status'] == 'success':
            return data['lat'], data['lon']
    except Exception as e:
        st.error(f"Could not auto-locate: {e}")
    return 42.36, -71.05  # Default to Boston if failed


# [LAMBDA] Lambda function to calculate price based on distance
# Base price $50 + $0.12 per mile
calculate_price = lambda dist: 50 + (dist * 0.12)


def get_real_gate(ident, target_date_str=None):
    """
    Fetches real-time gate info from FlightAware AeroAPI.
    Matches the specific departure date if provided.
    Costs 1 Credit per call.
    """
    if not API_FLIGHTAWARE_KEY:
        return f"{random.choice(['A', 'B', 'C'])}{random.randint(1, 50)}", "[Simulated - No API Key]"

    try:
        # Request context: Future flights need "start" parameter usually, or just parse list.
        # AeroAPI /flights/{ident} returns list of flights ~recent window.
        url = f"https://aeroapi.flightaware.com/aeroapi/flights/{ident}"
        headers = {"x-apikey": API_FLIGHTAWARE_KEY}

        # Add basic time params to broaden search if needed, but defaults usually cover +/- 2 days.
        print(f"DEBUG: Calling FlightAware API for {ident}...")
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            if 'flights' in data and data['flights']:
                flights = data['flights']
                print(f"DEBUG: Found {len(flights)} flights for {ident}. Filtering...")

                # Logic: Find flight matching target_date (CLOSEST MATCH)
                if target_date_str:
                    # # target_date_str is "2025-12-08 11:30" (Local time from Google Flights)
                    # We will parse it and compares to FlightAware UTC
                    try:
                        # Approximation: Treat Google Flights as naive (or UTC-ish for comparison)
                        target_dt = datetime.strptime(target_date_str, "%Y-%m-%d %H:%M")
                    except:
                        target_dt = None

                    best_match = None
                    min_diff = float('inf')

                    for f in flights:
                        fa_time_str = f.get('scheduled_out', '')
                        if fa_time_str:
                            # Parse "2025-12-08T11:15:00Z" -> datetime
                            # Truncate Z for simple naive comparison
                            try:
                                fa_dt = datetime.strptime(fa_time_str[:19], "%Y-%m-%dT%H:%M:%S")

                                if target_dt:
                                    # Calculate difference in hours
                                    diff = abs((fa_dt - target_dt).total_seconds())
                                    # Accept if within 24 hours
                                    if diff < min_diff and diff < 86400:
                                        min_diff = diff
                                        best_match = f
                            except:
                                continue

                    if best_match:
                        gate = best_match.get('departure', {}).get('gate', 'TBD')
                        status = best_match.get('status', 'Scheduled')
                        return gate, status

                    return None, None

                else:
                    # Fallback for no date provided (old behavior)
                    flight = flights[0]
                    gate = flight.get('departure', {}).get('gate', 'TBD')
                    status = flight.get('status', 'Scheduled')
                    return gate, status

    except Exception as e:
        print(f"FlightAware Error: {e}")
        return f"{random.choice(['A', 'B', 'C'])}{random.randint(1, 50)}", "[Simulated - API Error]"

    return f"{random.choice(['A', 'B', 'C'])}{random.randint(1, 50)}", "[Simulated - No Data]"


def generate_boarding_pass(cart_items):
    """
    Generates a text-based boarding pass for purchased items.
    Uses FlightAware Real-Time Data if available.
    """
    import random
    ticket_text = "✈️ SKYTREK BOARDING PASS ✈️\n"
    ticket_text += "=" * 40 + "\n\n"

    for i, item in enumerate(cart_items, 1):
        # [ROUNDTRIP] Check if item has multiple legs
        legs = item.get('legs', [])

        # If simplistic item (Route Planner) or Old Item, wrap it as 1 leg
        if not legs:
            legs = [{
                "route": item.get('route'),
                "dep_time": item.get('dep_time', 'Check Monitors'),
                "is_simulated": not item.get('is_real_flight', False)
            }]

        # Iterate legs to create tickets
        for leg_idx, leg in enumerate(legs):
            # Generate random seat/gate for realism
            seat = f"{random.randint(1, 30)}{random.choice(['A', 'B', 'C', 'D', 'E', 'F'])}"
            gate = f"{random.choice(['A', 'B', 'C'])}{random.randint(1, 50)}"
            status_line = ""

            # [REAL-TIME] Try to fetch real gate if it's a real flight
            if item.get('is_real_flight') and not leg.get('is_simulated'):
                try:
                    # Parse Identifier from Route: "BOS to LHR (UA 2136)"
                    raw_route = leg['route']
                    flight_ident = raw_route.split('(')[-1].replace(')', '').replace(' ', '')

                    # Fetch Data (Costs 1 Credit) using Leg Date
                    real_gate, real_status = get_real_gate(flight_ident, target_date_str=leg.get('dep_time'))

                    if real_gate:
                        gate = real_gate
                    if real_status:
                        status_line = f"STATUS:    {real_status.upper()}\n"
                except:
                    pass

            time_str = leg.get('dep_time', 'Check Monitors')

            ticket_text += f"FLIGHT TICKET #{i}-{leg_idx + 1}\n"
            ticket_text += f"PASSENGER: Guest Traveler\n"
            ticket_text += f"FLIGHT:    {leg['route']}\n"
            ticket_text += f"DEPART:    {time_str}\n"
            ticket_text += f"SEAT:      {seat}  |  GATE: {gate}\n"
            if status_line:
                ticket_text += status_line
            ticket_text += f"PRICE:     ${item['price']:.2f} (Total Bundle)\n"
            ticket_text += "-" * 40 + "\n\n"

    ticket_text += "=" * 40 + "\n"
    ticket_text += "Thank you for flying with SkyTrek!\n"

    return ticket_text


def init_session_state():
    """
    Initializes Streamlit session state variables.
    """
    if 'cart' not in st.session_state:
        st.session_state['cart'] = []


def main():
    init_session_state()
    st.title("✈️ SkyTrek: Interactive Flight Planner")

    # Load Data
    df = load_data()
    if df.empty:
        st.error("Data file not found or empty.")
        return
    # Comprehensive ISO 3166-1 alpha-2 mapping
    code_to_name = {
        'AF': 'Afghanistan', 'AX': 'Aland Islands', 'AL': 'Albania', 'DZ': 'Algeria', 'AS': 'American Samoa',
        'AD': 'Andorra', 'AO': 'Angola', 'AI': 'Anguilla', 'AQ': 'Antarctica', 'AG': 'Antigua and Barbuda',
        'AR': 'Argentina', 'AM': 'Armenia', 'AW': 'Aruba', 'AU': 'Australia', 'AT': 'Austria',
        'AZ': 'Azerbaijan', 'BS': 'Bahamas', 'BH': 'Bahrain', 'BD': 'Bangladesh', 'BB': 'Barbados',
        'BY': 'Belarus', 'BE': 'Belgium', 'BZ': 'Belize', 'BJ': 'Benin', 'BM': 'Bermuda',
        'BT': 'Bhutan', 'BO': 'Bolivia', 'BQ': 'Bonaire, Sint Eustatius and Saba', 'BA': 'Bosnia and Herzegovina',
        'BW': 'Botswana', 'BV': 'Bouvet Island', 'BR': 'Brazil', 'IO': 'British Indian Ocean Territory',
        'BN': 'Brunei Darussalam', 'BG': 'Bulgaria', 'BF': 'Burkina Faso', 'BI': 'Burundi', 'KH': 'Cambodia',
        'CM': 'Cameroon', 'CA': 'Canada', 'CV': 'Cape Verde', 'KY': 'Cayman Islands', 'CF': 'Central African Republic',
        'TD': 'Chad', 'CL': 'Chile', 'CN': 'China', 'CX': 'Christmas Island', 'CC': 'Cocos (Keeling) Islands',
        'CO': 'Colombia', 'KM': 'Comoros', 'CG': 'Congo', 'CD': 'Congo, Democratic Republic of the',
        'CK': 'Cook Islands', 'CR': 'Costa Rica', 'CI': 'Cote d\'Ivoire', 'HR': 'Croatia', 'CU': 'Cuba',
        'CW': 'Curacao', 'CY': 'Cyprus', 'CZ': 'Czech Republic', 'DK': 'Denmark', 'DJ': 'Djibouti',
        'DM': 'Dominica', 'DO': 'Dominican Republic', 'EC': 'Ecuador', 'EG': 'Egypt', 'SV': 'El Salvador',
        'GQ': 'Equatorial Guinea', 'ER': 'Eritrea', 'EE': 'Estonia', 'ET': 'Ethiopia',
        'FK': 'Falkland Islands (Malvinas)',
        'FO': 'Faroe Islands', 'FJ': 'Fiji', 'FI': 'Finland', 'FR': 'France', 'GF': 'French Guiana',
        'PF': 'French Polynesia', 'TF': 'French Southern Territories', 'GA': 'Gabon', 'GM': 'Gambia',
        'GE': 'Georgia', 'DE': 'Germany', 'GH': 'Ghana', 'GI': 'Gibraltar', 'GR': 'Greece',
        'GL': 'Greenland', 'GD': 'Grenada', 'GP': 'Guadeloupe', 'GU': 'Guam', 'GT': 'Guatemala',
        'GG': 'Guernsey', 'GN': 'Guinea', 'GW': 'Guinea-Bissau', 'GY': 'Guyana', 'HT': 'Haiti',
        'HM': 'Heard Island and McDonald Islands', 'VA': 'Holy See (Vatican City State)', 'HN': 'Honduras',
        'HK': 'Hong Kong', 'HU': 'Hungary', 'IS': 'Iceland', 'IN': 'India', 'ID': 'Indonesia',
        'IR': 'Iran, Islamic Republic of', 'IQ': 'Iraq', 'IE': 'Ireland', 'IM': 'Isle of Man', 'IL': 'Israel',
        'IT': 'Italy', 'JM': 'Jamaica', 'JP': 'Japan', 'JE': 'Jersey', 'JO': 'Jordan', 'KZ': 'Kazakhstan',
        'KE': 'Kenya', 'KI': 'Kiribati', 'KP': 'Korea, Democratic People\'s Republic of', 'KR': 'South Korea',
        'KW': 'Kuwait', 'KG': 'Kyrgyzstan', 'LA': 'Lao People\'s Democratic Republic', 'LV': 'Latvia',
        'LB': 'Lebanon', 'LS': 'Lesotho', 'LR': 'Liberia', 'LY': 'Libya', 'LI': 'Liechtenstein',
        'LT': 'Lithuania', 'LU': 'Luxembourg', 'MO': 'Macao', 'MK': 'Macedonia, the Former Yugoslav Republic of',
        'MG': 'Madagascar', 'MW': 'Malawi', 'MY': 'Malaysia', 'MV': 'Maldives', 'ML': 'Mali',
        'MT': 'Malta', 'MH': 'Marshall Islands', 'MQ': 'Martinique', 'MR': 'Mauritania', 'MU': 'Mauritius',
        'YT': 'Mayotte', 'MX': 'Mexico', 'FM': 'Micronesia, Federated States of', 'MD': 'Moldova, Republic of',
        'MC': 'Monaco', 'MN': 'Mongolia', 'ME': 'Montenegro', 'MS': 'Montserrat', 'MA': 'Morocco',
        'MZ': 'Mozambique', 'MM': 'Myanmar', 'NA': 'Namibia', 'NR': 'Nauru', 'NP': 'Nepal',
        'NL': 'Netherlands', 'NC': 'New Caledonia', 'NZ': 'New Zealand', 'NI': 'Nicaragua', 'NE': 'Niger',
        'NG': 'Nigeria', 'NU': 'Niue', 'NF': 'Norfolk Island', 'MP': 'Northern Mariana Islands', 'NO': 'Norway',
        'OM': 'Oman', 'PK': 'Pakistan', 'PW': 'Palau', 'PS': 'Palestine, State of', 'PA': 'Panama',
        'PG': 'Papua New Guinea', 'PY': 'Paraguay', 'PE': 'Peru', 'PH': 'Philippines', 'PN': 'Pitcairn',
        'PL': 'Poland', 'PT': 'Portugal', 'PR': 'Puerto Rico', 'QA': 'Qatar', 'RE': 'Reunion',
        'RO': 'Romania', 'RU': 'Russia', 'RW': 'Rwanda', 'BL': 'Saint Barthelemy',
        'SH': 'Saint Helena, Ascension and Tristan da Cunha', 'KN': 'Saint Kitts and Nevis', 'LC': 'Saint Lucia',
        'MF': 'Saint Martin (French part)', 'PM': 'Saint Pierre and Miquelon', 'VC': 'Saint Vincent and the Grenadines',
        'WS': 'Samoa', 'SM': 'San Marino', 'ST': 'Sao Tome and Principe', 'SA': 'Saudi Arabia', 'SN': 'Senegal',
        'RS': 'Serbia', 'SC': 'Seychelles', 'SL': 'Sierra Leone', 'SG': 'Singapore', 'SX': 'Sint Maarten (Dutch part)',
        'SK': 'Slovakia', 'SI': 'Slovenia', 'SB': 'Solomon Islands', 'SO': 'Somalia', 'ZA': 'South Africa',
        'GS': 'South Georgia and the South Sandwich Islands', 'SS': 'South Sudan', 'ES': 'Spain', 'LK': 'Sri Lanka',
        'SD': 'Sudan', 'SR': 'Suriname', 'SJ': 'Svalbard and Jan Mayen', 'SZ': 'Swaziland', 'SE': 'Sweden',
        'CH': 'Switzerland', 'SY': 'Syrian Arab Republic', 'TW': 'Taiwan, Province of China', 'TJ': 'Tajikistan',
        'TZ': 'Tanzania, United Republic of', 'TH': 'Thailand', 'TL': 'Timor-Leste', 'TG': 'Togo', 'TK': 'Tokelau',
        'TO': 'Tonga', 'TT': 'Trinidad and Tobago', 'TN': 'Tunisia', 'TR': 'Turkey', 'TM': 'Turkmenistan',
        'TC': 'Turks and Caicos Islands', 'TV': 'Tuvalu', 'UG': 'Uganda', 'UA': 'Ukraine', 'AE': 'United Arab Emirates',
        'GB': 'United Kingdom', 'US': 'United States', 'UM': 'United States Minor Outlying Islands', 'UY': 'Uruguay',
        'UZ': 'Uzbekistan', 'VU': 'Vanuatu', 'VE': 'Venezuela, Bolivarian Republic of', 'VN': 'Viet Nam',
        'VG': 'Virgin Islands, British', 'VI': 'Virgin Islands, U.S.', 'WF': 'Wallis and Futuna',
        'EH': 'Western Sahara',
        'YE': 'Yemen', 'ZM': 'Zambia', 'ZW': 'Zimbabwe'
    }

    def get_country_name(code):
        # [DICTMETHOD] Method 1: .get()
        return code_to_name.get(code, code)

    # Sidebar Navigation -
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Airport Explorer", "Route Planner", "Real-Time Booking", "My Cart",
                                      "Data Insights"])

    st.sidebar.markdown("---")
    st.sidebar.info(f"**Total Airports:** {len(df)}")

    # ---------------------------------------------------------
    # PAGE: HOME (Welcome Page)
    # ---------------------------------------------------------
    if page == "Home":
        st.header("✈️ Welcome to SkyTrek!")
        st.subheader("Your All-in-One Flight Planning Dashboard")

        st.markdown(""" 
        SkyTrek is designed to make flight planning seamless, fun, and data-driven. 
        Whether you are a traveler looking for tickets or an aviation enthusiast tracking planes, we've got you covered.
        """)

        st.divider()

        # Feature Overview
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 🌍 Explore")
            st.write(
                "Browse our database of **50,000+ airports** globally. Filter by country, state, and visualize them on an interactive map.")

        with col2:
            st.markdown("### 📏 Plan & Book")
            st.write(
                "Calculate flight distances and estimate costs. Connect to **Google Flights (SerpApi)** for real-time ticket booking.")

        with col3:
            st.markdown("### 📊 Analyze")
            st.write(
                "Explore **Data Insights** to see airport density heatmaps, top country statistics, and facility breakdowns.")

        st.divider()

        st.info("👈 **Get Started:** Use the Sidebar Menu to navigate to different tools!")

    # ---------------------------------------------------------
    # PAGE 1: AIRPORT EXPLORER
    # ---------------------------------------------------------
    elif page == "Airport Explorer":
        st.header("🌍 Explore Airports")
        st.write("Filter airports by Country and State, then view them on the map.")

        # Using a temporary column for efficient exact matching
        df['search_label'] = df['city'] + " (" + df['state'] + "), " + df['country']

        # Get unique sorted options
        search_options = sorted(list(df['search_label'].unique()))

        # [CONTROLS] Top Row Buttons (Reset + Find Nearest)
        c_reset, c_near = st.columns([1, 1.5])

        with c_reset:
            if st.button("🔄 Reset View (Show All)"):
                st.session_state['sb_search'] = ""
                st.session_state['sb_country'] = "All"
                st.session_state['sb_state'] = "All"
                st.session_state['show_nearest'] = False
                st.rerun()

        with c_near:
            if st.button("📍 Find Nearest (Auto-Locate)"):
                with st.spinner("Locating..."):
                    lat, lon = get_user_location_by_ip()
                    st.session_state['user_lat'] = lat
                    st.session_state['user_lon'] = lon
                    st.session_state['show_nearest'] = True

        # User Selection
        search_selection = st.selectbox("Search by City (Type to get suggestions):", [""] + search_options,
                                        key="sb_search")

        col1, col2 = st.columns(2)

        # [FILTER2] Hierarchical Filtering: Country -> State
        with col1:
            unique_countries = df['country'].unique()
            # [SORT] Sorting the list of countries alphabetically by name
            sorted_countries = sorted(unique_countries, key=lambda x: get_country_name(x))
            selected_country = st.selectbox(
                "Select Country",
                ["All"] + sorted_countries,
                format_func=lambda x: "All Countries" if x == "All" else get_country_name(x),
                key="sb_country"
            )

        with col2:
            if selected_country != "All":
                country_states = sorted(df[df['country'] == selected_country]['state'].unique())
                selected_state = st.selectbox("Select State/Region", ["All"] + country_states, key="sb_state")
            else:
                selected_state = "All"
                st.selectbox("Select State/Region", ["Select a Country first"], disabled=True)

        # Apply Filters
        filtered_df = df.copy()

        # 1. Apply Search Filter
        if search_selection:
            filtered_df = filtered_df[filtered_df['search_label'] == search_selection]
        else:
            if selected_country != "All":
                filtered_df = filtered_df[filtered_df['country'] == selected_country]
                if selected_state != "All":
                    filtered_df = filtered_df[filtered_df['state'] == selected_state]

        # [ST2] Slider for Altitude
        if not filtered_df.empty:
            min_alt, max_alt = get_min_max_altitude(df)
            altitude_range = st.slider("Filter by Altitude (ft)", min_alt, max_alt, (min_alt, max_alt))
            filtered_df = filtered_df[
                (filtered_df['altitude'] >= altitude_range[0]) &
                (filtered_df['altitude'] <= altitude_range[1])
                ]

        # --- Calculate Nearest (If Active) --
        nearest_airports = pd.DataFrame()
        curr_lat, curr_lon = 42.36, -71.05

        if st.session_state.get('show_nearest'):
            curr_lat = st.session_state.get('user_lat', 42.36)
            curr_lon = st.session_state.get('user_lon', -71.05)

            df_calc = df.copy()
            df_calc['distance_miles'] = calculate_distance(curr_lat, curr_lon, df_calc['lat'], df_calc['lon'])
            nearest_airports = df_calc.sort_values('distance_miles').head(3)

            st.success(f"📍 Location Found: {curr_lat}, {curr_lon}")
            c1, c2, c3 = st.columns(3)
            for idx, (i, row) in enumerate(nearest_airports.iterrows()):
                with [c1, c2, c3][idx]:
                    st.info(f"✈️ **{row['iata']}**\n\n{row['distance_miles']:.1f} mi")

        # --- UNIFIED MAP ---
        if st.session_state.get('show_nearest') and not nearest_airports.empty:
            st.subheader("Results: Best Airports Near You")
            view_state = pdk.ViewState(latitude=curr_lat, longitude=curr_lon, zoom=7)
            layers = [
                pdk.Layer(
                    'ScatterplotLayer',
                    data=pd.DataFrame({'lat': [curr_lat], 'lon': [curr_lon]}),
                    get_position='[lon, lat]', get_color='[0, 0, 255, 200]', get_radius=8000,
                ),
                pdk.Layer(
                    'ScatterplotLayer',
                    data=nearest_airports,
                    get_position='[lon, lat]', get_color='[255, 0, 0, 200]', get_radius=5000, pickable=True,
                ),
                pdk.Layer(
                    "ArcLayer",
                    data=nearest_airports,
                    get_source_position=[curr_lon, curr_lat],
                    get_target_position="[lon, lat]",
                    get_source_color=[0, 0, 255, 100],
                    # [MAP] Detailed Map
                    get_width=4
                )
            ]
            st.pydeck_chart(pdk.Deck(map_style=None, initial_view_state=view_state, layers=layers,
                                     tooltip={"text": "{name} ({iata})\n{distance_miles} mi"}))

        elif not filtered_df.empty:
            st.metric("Airports Showing", len(filtered_df))
            mid_lat = filtered_df['lat'].mean()
            mid_lon = filtered_df['lon'].mean()
            zoom_level = 2
            if selected_state != "All" or search_selection:
                zoom_level = 5

            st.pydeck_chart(pdk.Deck(
                map_style=None,
                initial_view_state=pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=zoom_level),
                layers=[
                    pdk.Layer(
                        'ScatterplotLayer',
                        data=filtered_df,
                        get_position='[lon, lat]',
                        get_color='[200, 30, 0, 160]',
                        get_radius=2000 if zoom_level < 4 else 1000,
                        pickable=True,
                    ),
                ],
                tooltip={"text": "{name} ({iata})\n{city}, {country}"}
            ))
        else:
            st.warning("No airports found.")

    # ---------------------------------------------------------
    # PAGE 2: ROUTE PLANNER
    # ---------------------------------------------------------
    elif page == "Route Planner":
        st.header("🗺️ Route Planner")
        st.write("Plan a trip between two airports.")

        def airport_selection(label_prefix):
            """
            Helper to create hierarchical selection widgets for an airport.
            """
            st.subheader(f"{label_prefix} Selection")

            c1, c2, c3 = st.columns(3)

            with c1:
                # Prepare options sorted by "Full Name"
                country_options = sorted(df['country'].unique(), key=lambda x: get_country_name(x))

                # Use format_func to show full name
                country = st.selectbox(
                    f"{label_prefix} Country",
                    country_options,
                    format_func=lambda x: get_country_name(x),
                    # [FUNCCALL2] Function called for different purpose/location
                    key=f"{label_prefix}_country"
                )

            with c2:
                states = sorted(df[df['country'] == country]['state'].unique())
                state = st.selectbox(f"{label_prefix} State", ["All"] + states, key=f"{label_prefix}_state")

            with c3:
                # Filter airports based on country and state
                airports_df = df[df['country'] == country]
                if state != "All":
                    airports_df = airports_df[airports_df['state'] == state]

                # Use DataFrame INDEX as the unique identifier
                options = airports_df.index.tolist()

                # Helper to format the display
                def format_option(idx):
                    row = df.loc[idx]
                    return f"{row['city']} - {row['name']} ({row['iata']})"

                if not options:
                    st.warning("No airports found")
                    selection_idx = None
                else:
                    # Sort options by the display string
                    options.sort(key=format_option)
                    selection_idx = st.selectbox(f"{label_prefix} Airport", options, format_func=format_option,
                                                 key=f"{label_prefix}_airport")

            return selection_idx

        # Origin Selection
        origin_idx = airport_selection("Origin")
        st.markdown("---")
        # Destination Selection
        dest_idx = airport_selection("Destination")

        if origin_idx is not None and dest_idx is not None and origin_idx != dest_idx:
            # Get airport data directly using the unique index
            origin_data = df.loc[origin_idx]
            dest_data = df.loc[dest_idx]

            origin_iata = origin_data['iata']
            dest_iata = dest_data['iata']

            # Passenger Inputs
            st.subheader("Passenger Details")
            c1, c2 = st.columns(2)
            with c1:
                adults = st.number_input("Adults (18+)", min_value=1, value=1, step=1)
            with c2:
                children = st.number_input("Children", min_value=0, value=0, step=1)

            # [FUNCCALL2] Call the distance function
            dist = calculate_distance(origin_data['lat'], origin_data['lon'],
                                      dest_data['lat'], dest_data['lon'])

            # Calculate Price per person
            base_price = calculate_price(dist)

            # Total Price Calculation
            # Assuming Child price is same as Adult for simplicity, or we could discount it.
            # Let's do a simple discount: Children are 70% of price
            child_price = base_price * 0.7
            total_price = (adults * base_price) + (children * child_price)

            # Display Flight Info
            st.success(f"✈️ Flight from {origin_data['city']} to {dest_data['city']}")

            m1, m2, m3 = st.columns(3)
            m1.metric("Distance", f"{int(dist)} miles")
            m2.metric("Price (Adult)", f"${base_price:.2f}")
            m3.metric("Flight Time", f"{dist / 500:.1f} hrs")

            st.info(f"**Total Price:** ${total_price:.2f} ({adults} Adults, {children} Children)")

            # [MAP] ArcLayer to show the route
            layer = pdk.Layer(
                "ArcLayer",
                data=[{
                    "source": [origin_data['lon'], origin_data['lat']],
                    "target": [dest_data['lon'], dest_data['lat']],
                    "name": f"{origin_iata} -> {dest_iata}"
                }],
                get_source_position="source",
                get_target_position="target",
                get_width=5,
                get_source_color=[0, 255, 0, 160],
                get_target_color=[255, 0, 0, 160],
            )
            view_state = pdk.ViewState(latitude=(origin_data['lat'] + dest_data['lat']) / 2,
                                       longitude=(origin_data['lon'] + dest_data['lon']) / 2, zoom=1)
            # Fix: Removed map_style here too
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, map_style=None))



        elif origin_idx == dest_idx and origin_idx is not None:
            st.error("Origin and Destination must be different.")
    # ---------------------------------------------------------
    # PAGE: REAL-TIME BOOKING (SerpApi Google Flights API)
    # ---------------------------------------------------------
    elif page == "Real-Time Booking":
        st.header("✈️ Real-Time Flight Search")
        st.caption("Powered by Google Flights (via SerpApi)")

        # 1. API Credentials
        with st.expander("API Credentials", expanded=True):
            st.info("Enter your SerpApi Key below.")
            #[ST3]
            api_key = st.text_input("SerpApi Key", value=API_SERPAPI_KEY, type="password")

        # 2. Search Inputs
        st.subheader("Find a Flight")
        col1, col2, col3 = st.columns(3)
        origin = col1.text_input("From (IATA)", value="BOS", help="e.g. JFK, LHR, CDG").upper()
        dest = col2.text_input("To (IATA)", value="LHR", help="e.g. LAX, DXB, HND").upper()
        c3_1, c3_2 = col3.columns(2)
        dept_date = c3_1.date_input("Departure")
        return_date = c3_2.date_input("Return (Optional)", value=None)

        # 3. Search Logic
        if st.button("Search Live Flights 🔎"):
            if not api_key:
                st.error("⚠️ Please enter your SerpApi Key above.")
            else:
                try:
                    with st.spinner("Searching Google Flights..."):
                        if GoogleSearch is None:
                            st.error("SerpApi library not found. Please install `google-search-results`.")
                            st.stop()

                        params = {
                            "engine": "google_flights",
                            "departure_id": origin,
                            "arrival_id": dest,
                            "outbound_date": str(dept_date),
                            "currency": "USD",
                            "hl": "en",
                            "api_key": api_key
                        }
                        if return_date:
                            params["return_date"] = str(return_date)
                            params["type"] = "1"  # Round Trip
                        else:
                            params["type"] = "2"  # One Way

                        search = GoogleSearch(params)
                        results = search.get_dict()

                        # [DEBUG] Check for API errors
                        if "error" in results:
                            st.error(f"SerpApi Error: {results['error']}")
                        else:
                            # Store in session state
                            st.session_state['serpapi_data'] = results.get('best_flights', [])

                            if not st.session_state['serpapi_data'] and 'other_flights' in results:
                                st.session_state['serpapi_data'] = results.get('other_flights', [])

                except Exception as e:
                    st.error(f"SerpApi Error: {e}")

        # 4. Display Results
        if 'serpapi_data' in st.session_state:
            flights = st.session_state['serpapi_data']

            if not flights:
                st.warning("No flights found.")
            else:
                st.success(f"Found {len(flights)} top options:")

                for idx, offer in enumerate(flights):
                    flight_id = f"serp_{idx}"
                    price_val = offer.get('price', 0)

                    # Parse Legs (SerpApi 'flights_cluster' structure?? No, 'flights' list)
                    # For simple results, 'flights' is a list of segments
                    segments = offer.get('flights', [])
                    legs_data = []

                    for seg in segments:
                        dep_port = seg.get('departure_airport', {})
                        arr_port = seg.get('arrival_airport', {})
                        time_str = dep_port.get('time', 'Unknown')  # "2025-12-08 10:00"

                        legs_data.append({
                            "route": f"{dep_port.get('id', origin)} to {arr_port.get('id', dest)} ({seg.get('airline')} {seg.get('flight_number')})",
                            "dep_time": time_str,
                            "carrier": seg.get('airline_codes', ['XX'])[0],
                            "number": seg.get('flight_number', '000')
                        })

                    # Display
                    if not legs_data: continue

                    route_summary = legs_data[0]['route']
                    if len(legs_data) > 1:
                        route_summary += f" + Connections/Return ({len(legs_data)} legs)"

                    with st.container():
                        c_icon, c_info, c_price = st.columns([1, 4, 2])
                        with c_icon:
                            st.markdown("✈️")

                        with c_info:
                            for leg in legs_data:
                                st.write(f"**{leg['route']}** | {leg['dep_time']}")

                        with c_price:
                            st.write(f"## **${price_val}**")
                            if st.button("Book 🛒", key=f"book_{flight_id}"):
                                st.session_state['cart'].append({
                                    "display_route": route_summary,
                                    "legs": legs_data,
                                    "price": float(price_val),
                                    "base_price": float(price_val),
                                    "distance": 0,
                                    "adults": 1,
                                    "children": 0,
                                    "is_real_flight": True,
                                    "route": route_summary
                                })
                                st.balloons()
                                st.success("Added!")
                        st.divider()

    # ---------------------------------------------------------
    # PAGE 3: MY CART
    # ---------------------------------------------------------
    elif page == "My Cart":
        st.header("🛒 My Shopping Cart")

        if st.session_state.get('checkout_success'):
            st.balloons()
            st.success("Thank you for your purchase! ✈️ Your tickets have been booked.")

            # [DOWNLOAD] Boarding Pass Button
            if 'last_ticket' in st.session_state:
                st.download_button(
                    label="🎟️ Download Boarding Pass",
                    data=st.session_state['last_ticket'],
                    file_name="SkyTrek_Boarding_Pass.txt",
                    mime="text/plain"
                )

            st.session_state['checkout_success'] = False

        if not st.session_state['cart']:
            st.info("Your cart is empty. Go to Route Planner to add flights!")
        else:
            # [ITERLOOP] Iterate through cart to display items and calculate total
            total_cost = 0

            # Use a container to display items
            for i, item in enumerate(st.session_state['cart']):
                with st.container():
                    st.markdown(f"### Flight {i + 1}: {item['route']}")

                    c1, c2, c3, c4 = st.columns([2, 2, 2, 1])

                    with c1:
                        # Editable Adult Count
                        new_adults = st.number_input(f"Adults (18+)", min_value=1, value=item['adults'],
                                                     key=f"adults_{i}")

                    with c2:
                        # Editable Child Count
                        new_children = st.number_input(f"Children", min_value=0, value=item['children'],
                                                       key=f"children_{i}")

                    # Hybrid Logic: Check if it's a "Real" flight or "Simulated"
                    if item.get('is_real_flight'):
                        # Use the fixed base price from the API
                        base_price = item['base_price']
                        dist = 0
                    else:
                        # Use the distance formula
                        dist = item['distance']
                        base_price = calculate_price(dist)

                    child_price = base_price * 0.7
                    new_total_price = (new_adults * base_price) + (new_children * child_price)

                    # Update session state to persist changes
                    st.session_state['cart'][i]['adults'] = new_adults
                    st.session_state['cart'][i]['children'] = new_children

                    with c3:
                        st.write(f"**Price:** ${new_total_price:.2f}")
                        if item.get('is_real_flight'):
                            st.caption("✅ Real-Time")
                        else:
                            st.caption(f"({int(dist)} miles)")
                        st.caption(f"**Tkts:** {new_adults + new_children}")

                    with c4:
                        st.write("")
                        if st.button("Remove", key=f"remove_{i}"):
                            # [DICTMETHOD] Method 2: .pop()
                            st.session_state['cart'].pop(i)
                            st.rerun()

                    st.markdown("---")
                    total_cost += new_total_price

            st.subheader(f"Total: ${total_cost:.2f}")

            # Checkout
            if st.button("Checkout & Pay"):
                # 1. Generate Ticket BEFORE clearing
                ticket_content = generate_boarding_pass(st.session_state['cart'])
                st.session_state['last_ticket'] = ticket_content

                # 2. Clear and Refresh
                st.session_state['cart'] = []
                st.session_state['checkout_success'] = True
                st.rerun()


    # ---------------------------------------------------------
    # PAGE: DATA INSIGHTS
    # ---------------------------------------------------------
    elif page == "Data Insights":
        st.header("📊 Data Analytics & Insights")

        # --- OPTION 1: GLOBAL HEATMAP USING SAME STYLE AS AIRPORT EXPLORER ---
        st.subheader("1. Global Airport Density Heatmap")
        st.write(
            "This map uses the same base style as Airport Explorer, "
            "but adds a density heat layer on top. Zoom in to see countries, "
            "states, and cities while still seeing where airports are concentrated."
        )

        # Center the view roughly on the world / your data
        mid_lat = df['lat'].mean()
        mid_lon = df['lon'].mean()

        view_state = pdk.ViewState(
            latitude=mid_lat,
            longitude=mid_lon,
            zoom=2,
            min_zoom=1,
            max_zoom=8,
            pitch=0,
            bearing=0,
        )

        # Heatmap layer (density)
        heatmap_layer = pdk.Layer(
            "HeatmapLayer",
            data=df,
            get_position='[lon, lat]',
            get_weight=1,
            radius_pixels=25,  # increase = smoother blob, decrease = sharper
            opacity=0.35,  # lower = more labels visible
            aggregation="SUM",
        )

        # Optional scatter dots so individual airports show when zoomed
        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position='[lon, lat]',
            get_radius=2000,
            get_color='[255, 255, 255, 160]',
            pickable=True,
        )

        st.pydeck_chart(
            pdk.Deck(
                map_style=None,  # ⬅️ SAME STYLE AS AIRPORT EXPLORER
                initial_view_state=view_state,
                layers=[heatmap_layer, scatter_layer],
                tooltip={"text": "Airport: {name}\nLocation: {city}, {country}"}
            )
        )

        # Custom Legend (Complexity Feature)
        st.markdown('''
            <style>
                .gradient-bar {
                    height: 20px; width: 100%;
                    background: linear-gradient(to right, rgb(65, 182, 196), rgb(255, 255, 204), rgb(128, 0, 38));
                    border-radius: 5px; border: 1px solid #ddd;
                }
                .labels { display: flex; justify-content: space-between; font-size: 12px; font-weight: bold; }
            </style>
            <div class="labels"><span>Low Density</span><span>Medium</span><span>High Density</span></div>
            <div class="gradient-bar"></div><br>
        ''', unsafe_allow_html=True)

        st.divider()

        # --- OPTION 2: PIE CHART (ACTIVE FACILITY TYPES ONLY) ---
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("2. Facility Types (Active Airports Only)")

            # Active types we want to show in the pie chart
            active_types = [
                "small_airport",
                "medium_airport",
                "large_airport",
                "heliport",
                "seaplane_base",
            ]

            # Filter to only active facility types (exclude closed)
            pie_data = df[df["type"].isin(active_types)]
            type_counts = pie_data["type"].value_counts()

            if type_counts.empty:
                st.warning("No active airports found for this dataset.")
            else:
                # Matplotlib pie chart
                fig2, ax2 = plt.subplots(figsize=(6, 4))

                # Slightly explode small slices to make them easier to see
                explode = [0.05 if v < type_counts.sum() * 0.1 else 0 for v in type_counts]
                # [CHART1] Pie Chart with custom explode and percentages
                wedges, texts, autotexts = ax2.pie(
                    type_counts,
                    autopct="%1.1f%%",
                    startangle=140,
                    explode=explode,
                    pctdistance=0.85,
                    textprops=dict(color="black"),
                )

                ax2.legend(
                    wedges,
                    type_counts.index,
                    title="Facility Type",
                    loc="center left",
                    bbox_to_anchor=(1, 0, 0.5, 1),
                )

                ax2.set_title("Distribution of Active Airport Types", fontsize=12)
                st.pyplot(fig2)

        # --- OPTION 3: BAR CHART (FACILITY TYPES BY COUNTRY, INCLUDING CLOSED) ---
        with c2:
            st.subheader("3. Airport Types by Country")

            # Let the user pick a country (use full name in the dropdown)
            country_options = sorted(df["country"].unique(), key=lambda x: get_country_name(x))
            default_idx = country_options.index("US") if "US" in country_options else 0

            selected_country_bar = st.selectbox(
                "Select Country:",
                country_options,
                index=default_idx,
                format_func=lambda x: get_country_name(x),
                key="country_facility_bar",
            )

            # Filter data to the selected country
            country_df = df[df["country"] == selected_country_bar]

            if country_df.empty:
                st.warning(f"No airport data available for {get_country_name(selected_country_bar)}.")
            else:
                # Types we care about in the bar chart (including closed)
                types_for_bar = [
                    "large_airport",
                    "medium_airport",
                    "small_airport",
                    "heliport",
                    "seaplane_base",
                    "closed",
                ]

                # Count types and reindex so all types appear (even if 0)
                type_counts_country = (
                    country_df["type"]
                    .value_counts()
                    .reindex(types_for_bar, fill_value=0)
                )

                fig3, ax3 = plt.subplots(figsize=(6, 4))
                # [CHART2] Horizontal Bar Chart showing facility type
                ax3.barh(
                    type_counts_country.index,
                    type_counts_country.values,
                    edgecolor="black",
                )
                ax3.set_title(
                    f"Airport Types in {get_country_name(selected_country_bar)}",
                    fontsize=12,
                    fontweight="bold",
                )
                ax3.set_xlabel("Number of Airports", fontsize=10)
                ax3.grid(axis="x", linestyle="--", alpha=0.5)

                st.pyplot(fig3)

        st.divider()

        # --- OPTION 4: PIVOT TABLE (#[PIVOTTABLE] + #[FILTER2] + #[ST1] + #[MAXMIN]) ---
        st.subheader("4. Detailed Breakdown by Region")

        # [ST1] Dropdown for Country Selection - IMPROVED
        all_countries = sorted(df['country'].unique(), key=lambda x: get_country_name(x))
        default_ix = all_countries.index('US') if 'US' in all_countries else 0

        # [FIX] Use format_func to show full names
        selected_country = st.selectbox("Select Country to Analyze:", all_countries, index=default_ix,
                                        format_func=lambda x: get_country_name(x))

        # [FILTER2] Filter by TWO conditions (Country AND Valid State)
        country_data = df[(df['country'] == selected_country) & (df['state'].notna())]

        if country_data.empty:
            st.warning(f"No state data available for {get_country_name(selected_country)}.")
        else:
            # [PIVOTTABLE] Pivot Table (Group By)
            pivot_df = country_data.groupby('state')[['name']].count()
            pivot_df.columns = ['Airport Count']  # [COLUMNS] Renaming column

            # Sort for better viewing
            pivot_df = pivot_df.sort_values('Airport Count', ascending=False)

            # Display Table
            st.dataframe(pivot_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)

            # [MAXMIN] Find largest value
            top_region = pivot_df.idxmax()[0]
            max_val = pivot_df.max()[0]
            st.success(
                f"**Insight:** The busiest region in {get_country_name(selected_country)} is **{top_region}** with **{max_val}** airports.")


if __name__ == "__main__":
    main()
