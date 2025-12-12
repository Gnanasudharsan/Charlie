"""
MBTA Real-Time Transit Dashboard with AI Chatbot + Location Integration
A comprehensive transit information system for Boston's public transportation.
Supports: Subway (Red, Orange, Blue, Green), Bus, Commuter Rail, and Ferry.
Includes AI-powered chatbot for natural language queries with Context Memory & Route Logic.
NEW: Location-based station detection and map integration.
UPDATED: Bold timing in chatbot responses for better readability.
"""

import os
import json
import math
import requests
from flask import Flask, jsonify, request, render_template
from dotenv import load_dotenv
from dateutil import parser
from datetime import datetime, timedelta
import pytz
from openai import OpenAI

# Load Environment Variables
load_dotenv()
MBTA_API_KEY = os.getenv("MBTA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Configuration
BASE_URL = "https://api-v3.mbta.com"
HEADERS = {"x-api-key": MBTA_API_KEY} if MBTA_API_KEY else {}
BOSTON_TZ = pytz.timezone('America/New_York')

# Store conversation history for context
conversation_history = []

# User's current location (updated via API)
user_location = {"lat": None, "lon": None, "station": None, "stop_id": None, "lines": []}

# Line Colors for UI
LINE_COLORS = {
    "Red": "#DA291C", "Orange": "#ED8B00", "Blue": "#003DA5",
    "Green-B": "#00843D", "Green-C": "#00843D", "Green-D": "#00843D", "Green-E": "#00843D",
    "Green": "#00843D", "Mattapan": "#DA291C", "Silver": "#7C878E",
    "Bus": "#FFC72C", "CR": "#80276C", "Ferry": "#008EAA"
}

# All Subway Routes
SUBWAY_ROUTES = ["Red", "Orange", "Blue", "Green-B", "Green-C", "Green-D", "Green-E", "Mattapan"]

# Commuter Rail Lines
COMMUTER_RAIL_LINES = [
    "CR-Fairmount", "CR-Fitchburg", "CR-Worcester", "CR-Franklin", "CR-Greenbush",
    "CR-Haverhill", "CR-Kingston", "CR-Lowell", "CR-Middleborough", "CR-Needham",
    "CR-Newburyport", "CR-Providence", "CR-Foxboro"
]

# ==================== COMPREHENSIVE STATION DATA WITH COORDINATES ====================
STATION_DATA = {
    # RED LINE
    "Alewife": {"id": "place-alfcl", "lat": 42.395428, "lon": -71.142483, "lines": ["Red"]},
    "Davis": {"id": "place-davis", "lat": 42.39674, "lon": -71.121815, "lines": ["Red"]},
    "Davis Square": {"id": "place-davis", "lat": 42.39674, "lon": -71.121815, "lines": ["Red"]},
    "Porter": {"id": "place-portr", "lat": 42.3884, "lon": -71.119149, "lines": ["Red"]},
    "Porter Square": {"id": "place-portr", "lat": 42.3884, "lon": -71.119149, "lines": ["Red"]},
    "Harvard": {"id": "place-harsq", "lat": 42.373362, "lon": -71.118956, "lines": ["Red"]},
    "Harvard Square": {"id": "place-harsq", "lat": 42.373362, "lon": -71.118956, "lines": ["Red"]},
    "Central": {"id": "place-cntsq", "lat": 42.365486, "lon": -71.103802, "lines": ["Red"]},
    "Central Square": {"id": "place-cntsq", "lat": 42.365486, "lon": -71.103802, "lines": ["Red"]},
    "Kendall/MIT": {"id": "place-knncl", "lat": 42.362491, "lon": -71.086176, "lines": ["Red"]},
    "Kendall": {"id": "place-knncl", "lat": 42.362491, "lon": -71.086176, "lines": ["Red"]},
    "MIT": {"id": "place-knncl", "lat": 42.362491, "lon": -71.086176, "lines": ["Red"]},
    "Charles/MGH": {"id": "place-chmnl", "lat": 42.361166, "lon": -71.070628, "lines": ["Red"]},
    "Park Street": {"id": "place-pktrm", "lat": 42.356395, "lon": -71.062424, "lines": ["Red", "Green-B", "Green-C", "Green-D", "Green-E"]},
    "Downtown Crossing": {"id": "place-dwnxg", "lat": 42.355518, "lon": -71.060225, "lines": ["Red", "Orange"]},
    "South Station": {"id": "place-sstat", "lat": 42.352271, "lon": -71.055242, "lines": ["Red"]},
    "Broadway": {"id": "place-brdwy", "lat": 42.342622, "lon": -71.056967, "lines": ["Red"]},
    "Andrew": {"id": "place-andrw", "lat": 42.330154, "lon": -71.057655, "lines": ["Red"]},
    "JFK/UMass": {"id": "place-jfk", "lat": 42.320685, "lon": -71.052391, "lines": ["Red"]},
    "JFK": {"id": "place-jfk", "lat": 42.320685, "lon": -71.052391, "lines": ["Red"]},
    "Savin Hill": {"id": "place-shmnl", "lat": 42.31129, "lon": -71.053331, "lines": ["Red"]},
    "Fields Corner": {"id": "place-fldcr", "lat": 42.300093, "lon": -71.061667, "lines": ["Red"]},
    "Shawmut": {"id": "place-smmnl", "lat": 42.29312, "lon": -71.065738, "lines": ["Red"]},
    "Ashmont": {"id": "place-asmnl", "lat": 42.284652, "lon": -71.064489, "lines": ["Red"]},
    "North Quincy": {"id": "place-nqncy", "lat": 42.275275, "lon": -71.029583, "lines": ["Red"]},
    "Wollaston": {"id": "place-wlsta", "lat": 42.2665, "lon": -71.0203, "lines": ["Red"]},
    "Quincy Center": {"id": "place-qnctr", "lat": 42.251809, "lon": -71.005409, "lines": ["Red"]},
    "Quincy Adams": {"id": "place-qamnl", "lat": 42.233391, "lon": -71.007153, "lines": ["Red"]},
    "Braintree": {"id": "place-brntn", "lat": 42.2078, "lon": -71.0011, "lines": ["Red"]},
    
    # ORANGE LINE
    "Oak Grove": {"id": "place-ogmnl", "lat": 42.43668, "lon": -71.071097, "lines": ["Orange"]},
    "Malden Center": {"id": "place-mlmnl", "lat": 42.426632, "lon": -71.07411, "lines": ["Orange"]},
    "Malden": {"id": "place-mlmnl", "lat": 42.426632, "lon": -71.07411, "lines": ["Orange"]},
    "Wellington": {"id": "place-welln", "lat": 42.40237, "lon": -71.077082, "lines": ["Orange"]},
    "Assembly": {"id": "place-astao", "lat": 42.392811, "lon": -71.077257, "lines": ["Orange"]},
    "Assembly Row": {"id": "place-astao", "lat": 42.392811, "lon": -71.077257, "lines": ["Orange"]},
    "Sullivan Square": {"id": "place-sull", "lat": 42.383975, "lon": -71.076994, "lines": ["Orange"]},
    "Sullivan": {"id": "place-sull", "lat": 42.383975, "lon": -71.076994, "lines": ["Orange"]},
    "Community College": {"id": "place-ccmnl", "lat": 42.373622, "lon": -71.069533, "lines": ["Orange"]},
    "North Station": {"id": "place-north", "lat": 42.365577, "lon": -71.06129, "lines": ["Orange", "Green-C", "Green-E"]},
    "Haymarket": {"id": "place-haecl", "lat": 42.363021, "lon": -71.05829, "lines": ["Orange", "Green-C", "Green-E"]},
    "State": {"id": "place-state", "lat": 42.358978, "lon": -71.057598, "lines": ["Orange", "Blue"]},
    "State Street": {"id": "place-state", "lat": 42.358978, "lon": -71.057598, "lines": ["Orange", "Blue"]},
    "Chinatown": {"id": "place-chncl", "lat": 42.352547, "lon": -71.062752, "lines": ["Orange"]},
    "Tufts Medical Center": {"id": "place-tumnl", "lat": 42.349662, "lon": -71.063917, "lines": ["Orange"]},
    "Tufts Medical": {"id": "place-tumnl", "lat": 42.349662, "lon": -71.063917, "lines": ["Orange"]},
    "Back Bay": {"id": "place-bbsta", "lat": 42.34735, "lon": -71.075727, "lines": ["Orange"]},
    "Mass Ave": {"id": "place-masta", "lat": 42.341512, "lon": -71.083423, "lines": ["Orange"]},
    "Massachusetts Avenue": {"id": "place-masta", "lat": 42.341512, "lon": -71.083423, "lines": ["Orange"]},
    "Ruggles": {"id": "place-rugg", "lat": 42.336377, "lon": -71.088961, "lines": ["Orange"]},
    "Roxbury Crossing": {"id": "place-rcmnl", "lat": 42.331397, "lon": -71.095451, "lines": ["Orange"]},
    "Jackson Square": {"id": "place-jaksn", "lat": 42.323132, "lon": -71.099592, "lines": ["Orange"]},
    "Stony Brook": {"id": "place-sbmnl", "lat": 42.317062, "lon": -71.104248, "lines": ["Orange"]},
    "Green Street": {"id": "place-grnst", "lat": 42.310525, "lon": -71.107414, "lines": ["Orange"]},
    "Forest Hills": {"id": "place-forhl", "lat": 42.300523, "lon": -71.113686, "lines": ["Orange"]},
    
    # BLUE LINE
    "Wonderland": {"id": "place-wondl", "lat": 42.41342, "lon": -70.991648, "lines": ["Blue"]},
    "Revere Beach": {"id": "place-rbmnl", "lat": 42.40784, "lon": -70.992533, "lines": ["Blue"]},
    "Beachmont": {"id": "place-bmmnl", "lat": 42.39754, "lon": -70.992319, "lines": ["Blue"]},
    "Suffolk Downs": {"id": "place-sdmnl", "lat": 42.39050, "lon": -70.997123, "lines": ["Blue"]},
    "Orient Heights": {"id": "place-orhte", "lat": 42.386867, "lon": -71.004736, "lines": ["Blue"]},
    "Wood Island": {"id": "place-wimnl", "lat": 42.379542, "lon": -71.022865, "lines": ["Blue"]},
    "Airport": {"id": "place-apts", "lat": 42.374262, "lon": -71.030395, "lines": ["Blue"]},
    "Logan Airport": {"id": "place-apts", "lat": 42.374262, "lon": -71.030395, "lines": ["Blue"]},
    "Maverick": {"id": "place-mvbcl", "lat": 42.36911, "lon": -71.03953, "lines": ["Blue"]},
    "Aquarium": {"id": "place-aqucl", "lat": 42.359784, "lon": -71.051652, "lines": ["Blue"]},
    "Government Center": {"id": "place-gover", "lat": 42.359705, "lon": -71.059215, "lines": ["Blue", "Green-B", "Green-C", "Green-D", "Green-E"]},
    "Govt Center": {"id": "place-gover", "lat": 42.359705, "lon": -71.059215, "lines": ["Blue", "Green-B", "Green-C", "Green-D", "Green-E"]},
    "Bowdoin": {"id": "place-bomnl", "lat": 42.361365, "lon": -71.062037, "lines": ["Blue"]},
    
    # GREEN LINE - COMMON
    "Lechmere": {"id": "place-lech", "lat": 42.370772, "lon": -71.076536, "lines": ["Green-D", "Green-E"]},
    "Science Park": {"id": "place-spmnl", "lat": 42.366664, "lon": -71.067666, "lines": ["Green-C", "Green-E"]},
    "Boylston": {"id": "place-boyls", "lat": 42.35302, "lon": -71.06459, "lines": ["Green-B", "Green-C", "Green-D", "Green-E"]},
    "Arlington": {"id": "place-arln", "lat": 42.351902, "lon": -71.070893, "lines": ["Green-B", "Green-C", "Green-D", "Green-E"]},
    "Copley": {"id": "place-coecl", "lat": 42.349974, "lon": -71.077447, "lines": ["Green-B", "Green-C", "Green-D", "Green-E"]},
    "Hynes Convention Center": {"id": "place-hymnl", "lat": 42.347888, "lon": -71.087903, "lines": ["Green-B", "Green-C", "Green-D"]},
    "Hynes": {"id": "place-hymnl", "lat": 42.347888, "lon": -71.087903, "lines": ["Green-B", "Green-C", "Green-D"]},
    "Kenmore": {"id": "place-kencl", "lat": 42.348949, "lon": -71.095169, "lines": ["Green-B", "Green-C", "Green-D"]},
    "Prudential": {"id": "place-prmnl", "lat": 42.345917, "lon": -71.081696, "lines": ["Green-E"]},
    "Pru": {"id": "place-prmnl", "lat": 42.345917, "lon": -71.081696, "lines": ["Green-E"]},
    "Symphony": {"id": "place-symcl", "lat": 42.342687, "lon": -71.085056, "lines": ["Green-E"]},
    
    # GREEN LINE - B BRANCH
    "Boston College": {"id": "place-lake", "lat": 42.340081, "lon": -71.166769, "lines": ["Green-B"]},
    "BC": {"id": "place-lake", "lat": 42.340081, "lon": -71.166769, "lines": ["Green-B"]},
    "South Street": {"id": "place-sougr", "lat": 42.3396, "lon": -71.1572, "lines": ["Green-B"]},
    "Chestnut Hill Ave": {"id": "place-chill", "lat": 42.338169, "lon": -71.15316, "lines": ["Green-B"]},
    "Chiswick Road": {"id": "place-chswk", "lat": 42.340302, "lon": -71.150711, "lines": ["Green-B"]},
    "Sutherland Road": {"id": "place-sthld", "lat": 42.341614, "lon": -71.146202, "lines": ["Green-B"]},
    "Washington Street": {"id": "place-wascm", "lat": 42.343864, "lon": -71.142853, "lines": ["Green-B"]},
    "Warren Street": {"id": "place-wrnst", "lat": 42.348343, "lon": -71.140457, "lines": ["Green-B"]},
    "Allston Street": {"id": "place-alsgr", "lat": 42.348701, "lon": -71.137955, "lines": ["Green-B"]},
    "Griggs Street": {"id": "place-grigg", "lat": 42.348545, "lon": -71.134949, "lines": ["Green-B"]},
    "Harvard Ave": {"id": "place-harvd", "lat": 42.350243, "lon": -71.131355, "lines": ["Green-B"]},
    "Harvard Avenue": {"id": "place-harvd", "lat": 42.350243, "lon": -71.131355, "lines": ["Green-B"]},
    "Packards Corner": {"id": "place-packr", "lat": 42.351967, "lon": -71.125031, "lines": ["Green-B"]},
    "Babcock Street": {"id": "place-babck", "lat": 42.35182, "lon": -71.12165, "lines": ["Green-B"]},
    "Pleasant Street": {"id": "place-plsgr", "lat": 42.351521, "lon": -71.117738, "lines": ["Green-B"]},
    "St Paul Street B": {"id": "place-stplb", "lat": 42.351340, "lon": -71.114685, "lines": ["Green-B"]},
    "BU West": {"id": "place-buwst", "lat": 42.350941, "lon": -71.113876, "lines": ["Green-B"]},
    "BU Central": {"id": "place-bucen", "lat": 42.350082, "lon": -71.106865, "lines": ["Green-B"]},
    "BU East": {"id": "place-buest", "lat": 42.349735, "lon": -71.103889, "lines": ["Green-B"]},
    "Blandford Street": {"id": "place-bland", "lat": 42.349293, "lon": -71.100258, "lines": ["Green-B"]},
    "Reservoir": {"id": "place-rsmnl", "lat": 42.335088, "lon": -71.148758, "lines": ["Green-B", "Green-C", "Green-D"]},

    # GREEN LINE - C BRANCH
    "Cleveland Circle": {"id": "place-clmnl", "lat": 42.336142, "lon": -71.149326, "lines": ["Green-C"]},
    "Englewood Ave": {"id": "place-engav", "lat": 42.336971, "lon": -71.145876, "lines": ["Green-C"]},
    "Dean Road": {"id": "place-denrd", "lat": 42.337807, "lon": -71.141853, "lines": ["Green-C"]},
    "Tappan Street": {"id": "place-tapst", "lat": 42.338459, "lon": -71.138702, "lines": ["Green-C"]},
    "Washington Square": {"id": "place-bcnwa", "lat": 42.339394, "lon": -71.13533, "lines": ["Green-C"]},
    "Fairbanks Street": {"id": "place-fbkst", "lat": 42.339725, "lon": -71.131073, "lines": ["Green-C"]},
    "Brandon Hall": {"id": "place-bndhl", "lat": 42.340023, "lon": -71.129082, "lines": ["Green-C"]},
    "Summit Ave": {"id": "place-sumav", "lat": 42.341002, "lon": -71.12561, "lines": ["Green-C"]},
    "Coolidge Corner": {"id": "place-cool", "lat": 42.342116, "lon": -71.121263, "lines": ["Green-C"]},
    "St Paul Street C": {"id": "place-stpul", "lat": 42.343327, "lon": -71.116997, "lines": ["Green-C"]},
    "St Paul Street": {"id": "place-stpul", "lat": 42.343327, "lon": -71.116997, "lines": ["Green-C"]},
    "Kent Street": {"id": "place-kntst", "lat": 42.344074, "lon": -71.114064, "lines": ["Green-C"]},
    "Hawes Street": {"id": "place-hwsst", "lat": 42.344906, "lon": -71.111145, "lines": ["Green-C"]},
    "St Marys Street": {"id": "place-smary", "lat": 42.345974, "lon": -71.107353, "lines": ["Green-C"]},

    # GREEN LINE - D BRANCH
    "Riverside": {"id": "place-river", "lat": 42.337352, "lon": -71.252685, "lines": ["Green-D"]},
    "Woodland": {"id": "place-woodl", "lat": 42.3328, "lon": -71.24305, "lines": ["Green-D"]},
    "Waban": {"id": "place-waban", "lat": 42.325845, "lon": -71.230609, "lines": ["Green-D"]},
    "Eliot": {"id": "place-eliot", "lat": 42.319023, "lon": -71.216713, "lines": ["Green-D"]},
    "Newton Highlands": {"id": "place-newtn", "lat": 42.322381, "lon": -71.205509, "lines": ["Green-D"]},
    "Newton Centre": {"id": "place-newto", "lat": 42.329443, "lon": -71.192413, "lines": ["Green-D"]},
    "Chestnut Hill": {"id": "place-chhil", "lat": 42.326653, "lon": -71.164699, "lines": ["Green-D"]},
    "Beaconsfield": {"id": "place-bcnfd", "lat": 42.335765, "lon": -71.140455, "lines": ["Green-D"]},
    "Brookline Hills": {"id": "place-brkhl", "lat": 42.331333, "lon": -71.126999, "lines": ["Green-D"]},
    "Brookline Village": {"id": "place-bvmnl", "lat": 42.332917, "lon": -71.11679, "lines": ["Green-D"]},
    "Longwood": {"id": "place-lngmd", "lat": 42.341145, "lon": -71.110451, "lines": ["Green-D"]},
    "Fenway": {"id": "place-fenwy", "lat": 42.345394, "lon": -71.104187, "lines": ["Green-D"]},

    # GREEN LINE - E BRANCH
    "Heath Street": {"id": "place-hsmnl", "lat": 42.328316, "lon": -71.110252, "lines": ["Green-E"]},
    "Back of the Hill": {"id": "place-bckhl", "lat": 42.330139, "lon": -71.111313, "lines": ["Green-E"]},
    "Riverway": {"id": "place-rvrwy", "lat": 42.331684, "lon": -71.111931, "lines": ["Green-E"]},
    "Mission Park": {"id": "place-mispk", "lat": 42.333195, "lon": -71.109756, "lines": ["Green-E"]},
    "Fenwood Road": {"id": "place-fenwd", "lat": 42.334182, "lon": -71.105667, "lines": ["Green-E"]},
    "Brigham Circle": {"id": "place-brmnl", "lat": 42.334377, "lon": -71.104167, "lines": ["Green-E"]},
    "Longwood Medical Area": {"id": "place-lngmd", "lat": 42.33596, "lon": -71.100052, "lines": ["Green-E"]},
    "MFA": {"id": "place-mfa", "lat": 42.337711, "lon": -71.095512, "lines": ["Green-E"]},
    "Museum of Fine Arts": {"id": "place-mfa", "lat": 42.337711, "lon": -71.095512, "lines": ["Green-E"]},
    "Northeastern": {"id": "place-nuniv", "lat": 42.340401, "lon": -71.088806, "lines": ["Green-E"]},
    "Northeastern University": {"id": "place-nuniv", "lat": 42.340401, "lon": -71.088806, "lines": ["Green-E"]},
    "NEU": {"id": "place-nuniv", "lat": 42.340401, "lon": -71.088806, "lines": ["Green-E"]},
    "Union Square": {"id": "place-unsqu", "lat": 42.377359, "lon": -71.094761, "lines": ["Green-D"]},
    "East Somerville": {"id": "place-esomr", "lat": 42.379482, "lon": -71.086625, "lines": ["Green-D"]},
    "Gilman Square": {"id": "place-gilmn", "lat": 42.387867, "lon": -71.096766, "lines": ["Green-D"]},
    "Magoun Square": {"id": "place-mgngl", "lat": 42.393171, "lon": -71.106046, "lines": ["Green-D"]},
    "Ball Square": {"id": "place-balsq", "lat": 42.399622, "lon": -71.110721, "lines": ["Green-D"]},
    "Medford/Tufts": {"id": "place-mdftf", "lat": 42.407975, "lon": -71.116865, "lines": ["Green-D"]},
    "Medford": {"id": "place-mdftf", "lat": 42.407975, "lon": -71.116865, "lines": ["Green-D"]},
    "Tufts": {"id": "place-mdftf", "lat": 42.407975, "lon": -71.116865, "lines": ["Green-D"]},
    
    # MATTAPAN TROLLEY
    "Mattapan": {"id": "place-matt", "lat": 42.267563, "lon": -71.092526, "lines": ["Mattapan"]},
    "Cedar Grove": {"id": "place-cedgr", "lat": 42.279629, "lon": -71.060394, "lines": ["Mattapan"]},
    "Butler": {"id": "place-butlr", "lat": 42.272429, "lon": -71.062519, "lines": ["Mattapan"]},
    "Milton": {"id": "place-miltt", "lat": 42.270349, "lon": -71.067266, "lines": ["Mattapan"]},
    "Central Avenue": {"id": "place-cenav", "lat": 42.270027, "lon": -71.073448, "lines": ["Mattapan"]},
    "Valley Road": {"id": "place-valrd", "lat": 42.268347, "lon": -71.081343, "lines": ["Mattapan"]},
    "Capen Street": {"id": "place-capst", "lat": 42.267563, "lon": -71.087338, "lines": ["Mattapan"]},
}

# Create STATION_TO_ID for backward compatibility
STATION_TO_ID = {name: data["id"] for name, data in STATION_DATA.items()}

VALID_STATIONS = list(STATION_DATA.keys())

# Station name aliases for typo correction
STATION_ALIASES = {
    "harverd": "Harvard", "harward": "Harvard", "havard": "Harvard",
    "alewhife": "Alewife", "alewif": "Alewife", "alwife": "Alewife",
    "dowtown": "Downtown Crossing", "dtx": "Downtown Crossing",
    "parkstreet": "Park Street", "park st": "Park Street",
    "southstation": "South Station", "south sta": "South Station",
    "northstation": "North Station", "north sta": "North Station",
    "foresthills": "Forest Hills", "forest hill": "Forest Hills",
    "oakgrove": "Oak Grove", "oak": "Oak Grove",
    "backbay": "Back Bay", "back": "Back Bay",
    "govtcenter": "Government Center", "gov center": "Government Center", "govcenter": "Government Center",
    "bc": "Boston College", "boston collge": "Boston College",
    "clevland": "Cleveland Circle", "cleveland": "Cleveland Circle",
    "riverside": "Riverside", "river side": "Riverside",
    "heath": "Heath Street", "heathstreet": "Heath Street",
    "harvard ave": "Harvard Avenue", "harvardave": "Harvard Avenue", "harvard av": "Harvard Avenue",
    "kendal": "Kendall/MIT", "kendall mit": "Kendall/MIT",
    "neu": "Northeastern", "northeastrn": "Northeastern",
    "pru": "Prudential", "prudentail": "Prudential",
    "hines": "Hynes Convention Center", "hynes": "Hynes Convention Center",
    "coolidge": "Coolidge Corner", "cooldige": "Coolidge Corner",
    "sullivan sq": "Sullivan Square", "sully": "Sullivan Square",
    "logan": "Airport", "logan airport": "Airport",
    "mfa": "MFA", "museum": "MFA",
    "tufts": "Tufts Medical Center", "tuftsmedical": "Tufts Medical Center",
    "assembly row": "Assembly", "assemblyrow": "Assembly",
    "union sq": "Union Square", "unionsquare": "Union Square",
    "medford tufts": "Medford/Tufts", "medford": "Medford/Tufts",
}

# Default routes based on destination (for simple lookups)
DEFAULT_ROUTES = {
    "Alewife": {"route": "Red", "dir": 1}, "Ashmont": {"route": "Red", "dir": 0},
    "Braintree": {"route": "Red", "dir": 0}, "JFK/UMass": {"route": "Red", "dir": 0},
    "Oak Grove": {"route": "Orange", "dir": 1}, "Forest Hills": {"route": "Orange", "dir": 0},
    "Wonderland": {"route": "Blue", "dir": 1}, "Bowdoin": {"route": "Blue", "dir": 0},
    "Boston College": {"route": "Green-B", "dir": 0}, "BC": {"route": "Green-B", "dir": 0},
    "Cleveland Circle": {"route": "Green-C", "dir": 0},
    "Riverside": {"route": "Green-D", "dir": 0}, "Union Square": {"route": "Green-D", "dir": 1},
    "Heath Street": {"route": "Green-E", "dir": 0}, "Medford/Tufts": {"route": "Green-E", "dir": 1},
    "Government Center": {"route": "Green-D", "dir": 1}, "Park Street": {"route": "Green-B", "dir": 1},
    "North Station": {"route": "Green-E", "dir": 1}, "Copley": {"route": "Green-B", "dir": 1},
    "Kenmore": {"route": "Green-B", "dir": 1}, "Lechmere": {"route": "Green-E", "dir": 1},
    "Downtown Crossing": {"route": "Red", "dir": 0}, "South Station": {"route": "Red", "dir": 0},
    "Harvard": {"route": "Red", "dir": 1}, "Haymarket": {"route": "Orange", "dir": 0},
    "State": {"route": "Orange", "dir": 0}, "Aquarium": {"route": "Blue", "dir": 0},
    "Airport": {"route": "Blue", "dir": 1}, "Mattapan": {"route": "Mattapan", "dir": 0},
}

# Route aliases
ROUTE_ALIASES = {
    "red": "Red", "red line": "Red", "redline": "Red",
    "orange": "Orange", "orange line": "Orange", "orangeline": "Orange",
    "blue": "Blue", "blue line": "Blue", "blueline": "Blue",
    "green": "Green-B", "green line": "Green-B", "greenline": "Green-B",
    "green b": "Green-B", "green-b": "Green-B", "greenb": "Green-B", "b line": "Green-B", "b train": "Green-B",
    "green c": "Green-C", "green-c": "Green-C", "greenc": "Green-C", "c line": "Green-C", "c train": "Green-C",
    "green d": "Green-D", "green-d": "Green-D", "greend": "Green-D", "d line": "Green-D", "d train": "Green-D",
    "green e": "Green-E", "green-e": "Green-E", "greene": "Green-E", "e line": "Green-E", "e train": "Green-E",
    "mattapan": "Mattapan", "mattapan trolley": "Mattapan", "mattapan line": "Mattapan",
    "silver": "741", "silver line": "741", "sl": "741", "sl1": "741", "sl2": "742", "sl3": "743",
}

# ==================== LOCATION FUNCTIONS ====================

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two GPS coordinates in meters."""
    R = 6371000  # Earth's radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c


def find_nearest_station(lat, lon, max_distance=2000):
    """Find the nearest MBTA station within max_distance meters."""
    nearest = None
    min_distance = float('inf')
    
    # Track unique stations by ID to avoid duplicates
    seen_ids = set()
    
    for name, data in STATION_DATA.items():
        if data["id"] in seen_ids:
            continue
        seen_ids.add(data["id"])
        
        distance = haversine_distance(lat, lon, data["lat"], data["lon"])
        if distance < min_distance and distance <= max_distance:
            min_distance = distance
            nearest = {
                "name": name,
                "id": data["id"],
                "distance": round(distance),
                "lines": data["lines"],
                "lat": data["lat"],
                "lon": data["lon"]
            }
    
    return nearest


# ==================== UTILITY FUNCTIONS ====================

def get_now():
    return datetime.now(BOSTON_TZ)

def format_time_diff(arrival_dt):
    now = get_now()
    if arrival_dt.tzinfo is None:
        arrival_dt = BOSTON_TZ.localize(arrival_dt)
    diff = (arrival_dt - now).total_seconds()
    minutes = int(diff / 60)
    if minutes < 0: return "Departed"
    elif minutes == 0: return "Arriving"
    elif minutes == 1: return "1 min"
    elif minutes < 60: return f"{minutes} min"
    else:
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours}h {mins}m" if mins else f"{hours}h"

def format_time(dt_str):
    if not dt_str: return None
    try:
        dt = parser.isoparse(dt_str)
        return dt.strftime("%-I:%M %p")
    except: return None

def normalize_station_name(name):
    if not name: return None
    lower_name = name.lower().strip()
    if lower_name in STATION_ALIASES:
        return STATION_ALIASES[lower_name]
    for station in VALID_STATIONS:
        if station.lower() == lower_name:
            return station
    for station in VALID_STATIONS:
        if lower_name in station.lower() or station.lower() in lower_name:
            return station
    return name.title()

def resolve_route(route_name):
    if not route_name: return None
    normalized = route_name.lower().strip()
    if normalized in ROUTE_ALIASES:
        return ROUTE_ALIASES[normalized]
    if route_name in SUBWAY_ROUTES:
        return route_name
    return None

def find_common_route(origin_name, dest_name):
    if not origin_name or not dest_name: return None
    
    origin_data = STATION_DATA.get(origin_name)
    dest_data = STATION_DATA.get(dest_name)
    
    if origin_data and dest_data:
        origin_lines = set(origin_data.get("lines", []))
        dest_lines = set(dest_data.get("lines", []))
        common = origin_lines.intersection(dest_lines)
        
        if len(common) == 1:
            return list(common)[0]
        elif len(common) > 1:
            for preferred in ["Red", "Orange", "Blue", "Green-D", "Green-E", "Green-B", "Green-C"]:
                if preferred in common:
                    return preferred
    
    return None


# ==================== CHATBOT FUNCTIONS ====================

def get_smart_intent(user_text):
    """Step 1: Understand what the user wants with Memory Logic and Location Awareness"""
    global conversation_history, user_location
    
    current_message = {"role": "user", "content": user_text}
    conversation_history.append(current_message)
    if len(conversation_history) > 10:
        conversation_history.pop(0)

    location_context = ""
    if user_location.get("station"):
        location_context = f"""
    USER'S CURRENT LOCATION: The user is currently at or near '{user_location['station']}' station.
    If the user asks about "next train" or "going to [destination]" WITHOUT specifying origin, 
    you should set origin to "USE_CURRENT_LOCATION" to indicate we should use their GPS location.
    """

    system_prompt = f"""You are an expert MBTA Transit Assistant. 
    
    CRITICAL RULE - MEMORY MERGE:
    The user might reply with just one word (e.g., "Orange").
    You MUST look at the previous messages.
    If the previous message had an Origin and Destination, YOU MUST COPY THEM to the new output.
    DO NOT output null for origin/destination if they were mentioned in the last 3 turns.
    
    LOCATION AWARENESS:
    {location_context}
    
    TRIP PLANNING DETECTION:
    If user asks HOW TO GET from one place to another, or asks for DIRECTIONS, ROUTE, or WAY TO GO,
    use intent "trip_planning". Examples:
    - "How do I get from Harvard to Airport?"
    - "What's the best way to go from Park Street to Forest Hills?"
    - "I'm at Ruggles, how can I reach Harvard?"
    - "Directions from South Station to Fenway"
    - "How to travel from Downtown to Assembly?"
    - "Route from Copley to Wonderland"
    - "I want to go from Central to Back Bay"
    - "Take me from Davis to Airport"
    
    OUTPUT JSON:
    {{
        "intent": "prediction" | "greeting" | "thanks" | "alerts" | "bus" | "commuter_rail" | "trip_planning" | "help" | "general",
        "origin": "Station Name (normalized) or USE_CURRENT_LOCATION if user didn't specify and we should use GPS",
        "destination": "Station Name (normalized)",
        "specific_route": "Green-B" (if mentioned OR inferred),
        "time_offset": 0,
        "bus_route": "66" (if bus),
        "commuter_line": "Worcester" (if commuter rail),
        "original_station_mentioned": "what user typed"
    }}
    
    VALID STATIONS: {json.dumps(VALID_STATIONS[:80])}
    """

    messages = [{"role": "system", "content": system_prompt}] + conversation_history

    try:
        if not client: return None
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.1
        )
        result = json.loads(response.choices[0].message.content)
        print(f"AI Intent: {result}")
        return result
    except Exception as e:
        print(f"OpenAI Error: {e}")
        return None

def generate_natural_response(user_text, system_data):
    """Step 3: Generate human response with BOLD TIMING for easy reading"""
    if not client: 
        return "I found the data, but I'm in basic mode. Check the JSON response."

    system_prompt = """
    You are a friendly, local Boston MBTA transit assistant. 
    I will provide you with specific REAL-TIME data found by the system.
    
    Your Goal: Answer the user's question naturally using the provided data.
    
    ═══════════════════════════════════════════════════════════════
    CRITICAL FORMATTING RULES - MAKE TIMING STAND OUT:
    ═══════════════════════════════════════════════════════════════
    
    1. ALWAYS make timing information BOLD using **double asterisks**:
       - Minutes away: **2 min**, **5 min**, **12 min**, **Arriving**
       - Clock times: **10:45 AM**, **3:30 PM**, **11:02 AM**
       - Duration: **25 min total**, **8 min walk**
    
    2. RESPONSE STRUCTURE for train predictions:
       - Start with brief context (location if auto-detected)
       - Show the NEXT train prominently with both time formats
       - List additional trains in a clear format
       - Keep it scannable and easy to read
    
    3. EXAMPLE FORMAT for predictions:
       
       🚇 From Harvard to Boston College on the Green-B:
       
       🔜 **Next train: 3 min** (at **10:42 AM**)
       
       Following trains:
       • **8 min** — **10:47 AM**
       • **15 min** — **10:54 AM**
       • **22 min** — **11:01 AM**
    
    4. If location was auto-detected, mention it:
       "📍 Since you're near **Harvard**..."
    
    ═══════════════════════════════════════════════════════════════
    TRIP PLANNING RESPONSE FORMAT (VERY IMPORTANT):
    ═══════════════════════════════════════════════════════════════
    
    When intent is "trip_planning" and routes are provided, you MUST show ALL routes.
    Format EXACTLY like this:
    
    🗺️ **Directions from [Origin] to [Destination]**
    
    I found **[X] routes** for you:
    
    ---
    
    **🥇 Route 1 (Recommended):** **[duration]** | [transfers] | 💰 [fare]
    📍 [depart_time] — [arrive_time]
    
    Steps:
    1. 🚶 Walk to [station] (**X min**)
    2. 🚇 Take **[Line Name]** toward [direction] (**X min**, X stops)
       → Board at **[station]**, exit at **[station]**
    3. 🔄 Transfer to **[Line Name]**
    4. 🚌 Take **Bus [number]** (**X min**, X stops)
    5. 🚶 Walk to destination (**X min**)
    
    ---
    
    **🥈 Route 2:** **[duration]** | [transfers] | 💰 [fare]
    📍 [time range]
    • [Line 1] → [Line 2] → ...
    
    ---
    
    **🥉 Route 3:** **[duration]** | [transfers] | 💰 [fare]
    📍 [time range]
    • [Line 1] → [Line 2] → ...
    
    (Continue for all routes provided)
    
    ---
    
    CRITICAL RULES FOR TRIP PLANNING:
    - Show ALL routes from the data (usually 3-6 routes)
    - Make DURATION bold and prominent: **42 mins**
    - Make LINE NAMES bold: **Green Line B**, **Bus 66**, **Orange Line**
    - Show the step_summary for quick overview: Green-B → Bus 66 → Orange Line
    - Include fare, transfers, and walk time for each route
    - The FIRST route should show detailed steps
    - Other routes can show summary (line sequence)
    - Use medals: 🥇 🥈 🥉 for top 3, then numbers for rest
    
    ═══════════════════════════════════════════════════════════════
    
    Other Guidelines:
    - Talk like a helpful human, not a robot
    - Be concise but conversational  
    - Use emojis: 🚇 subway, 🚌 bus, 🚂 commuter rail, 🚶 walk, 🔄 transfer, ⛴️ ferry
    - If no predictions, apologize and suggest alternatives
    - Do NOT invent data - only use what is provided
    
    THE MOST IMPORTANT THING: Show ALL routes and make timing/line names **BOLD**!
    """
    
    user_prompt = f"""
    User Input: "{user_text}"
    System Data Found: {json.dumps(system_data)}
    
    Write the response showing ALL routes with timing and line names in **bold**.
    For trip_planning, show every route from the data.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Generation Error: {e}")
        return "I found the data, but I'm having trouble phrasing it right now."

def get_bus_predictions(route_id, stop_query=None, dest_query=None):
    try:
        stops_resp = requests.get(f"{BASE_URL}/stops", headers=HEADERS,
            params={"filter[route]": route_id, "page[limit]": 100})
        stops = stops_resp.json().get('data', [])
        
        target_stop = None
        if stop_query:
            query_lower = stop_query.lower()
            for stop in stops:
                if query_lower in stop['attributes'].get('name', '').lower():
                    target_stop = stop
                    break
        
        if not target_stop:
            return [], None

        pred_resp = requests.get(f"{BASE_URL}/predictions", headers=HEADERS,
            params={
                "filter[stop]": target_stop['id'], 
                "filter[route]": route_id,
                "include": "trip",
                "sort": "arrival_time", 
                "page[limit]": 10
            })
        
        data = pred_resp.json()
        predictions = []
        
        included_trips = {item['id']: item for item in data.get('included', []) if item['type'] == 'trip'}
        
        for pred in data.get('data', []):
            trip_id = pred['relationships']['trip']['data']['id']
            trip_info = included_trips.get(trip_id, {})
            headsign = trip_info.get('attributes', {}).get('headsign', 'Unknown')
            
            if dest_query:
                if dest_query.lower() not in headsign.lower():
                    continue 

            arrival = pred['attributes'].get('arrival_time') or pred['attributes'].get('departure_time')
            if arrival:
                arrival_dt = parser.isoparse(arrival)
                predictions.append({
                    "time_away": format_time_diff(arrival_dt),
                    "arrival_time": format_time(arrival),
                    "headsign": headsign
                })
                
        return predictions, target_stop['attributes'].get('name', '')
    except Exception as e:
        print(f"Bus prediction error: {e}")
        return [], None


def get_commuter_rail_predictions(line_name, stop_query=None, dest_query=None):
    try:
        route_id = None
        for cr_line in COMMUTER_RAIL_LINES:
            if line_name.lower() in cr_line.lower():
                route_id = cr_line
                break
        if not route_id: return [], None
        
        stops_resp = requests.get(f"{BASE_URL}/stops", headers=HEADERS,
            params={"filter[route]": route_id, "page[limit]": 100})
        stops = stops_resp.json().get('data', [])
        
        target_stop = None
        if stop_query:
            query_lower = stop_query.lower()
            for stop in stops:
                if query_lower in stop['attributes'].get('name', '').lower():
                    target_stop = stop
                    break
        if not target_stop and stops: target_stop = stops[0]
        
        if target_stop:
            pred_resp = requests.get(f"{BASE_URL}/predictions", headers=HEADERS,
                params={
                    "filter[stop]": target_stop['id'], 
                    "filter[route]": route_id,
                    "include": "trip",
                    "sort": "arrival_time", 
                    "page[limit]": 10
                })
            
            data = pred_resp.json()
            predictions = []
            included_trips = {item['id']: item for item in data.get('included', []) if item['type'] == 'trip'}

            for pred in data.get('data', []):
                trip_id = pred['relationships']['trip']['data']['id']
                trip_info = included_trips.get(trip_id, {})
                headsign = trip_info.get('attributes', {}).get('headsign', 'Unknown')
                
                if dest_query and dest_query.lower() not in headsign.lower():
                    continue

                arrival = pred['attributes'].get('arrival_time') or pred['attributes'].get('departure_time')
                if arrival:
                    arrival_dt = parser.isoparse(arrival)
                    predictions.append({
                        "time_away": format_time_diff(arrival_dt),
                        "arrival_time": format_time(arrival),
                        "headsign": headsign
                    })
            return predictions, target_stop['attributes'].get('name', '')
        return [], None
    except Exception as e:
        print(f"Commuter rail error: {e}")
        return [], None


def get_alerts_for_route(route_id=None, limit=5):
    params = {"sort": "-severity", "page[limit]": limit}
    if route_id: params["filter[route]"] = route_id
    
    try:
        response = requests.get(f"{BASE_URL}/alerts", headers=HEADERS, params=params)
        data = response.json()
        alerts = []
        for alert in data.get('data', []):
            attrs = alert['attributes']
            alerts.append({
                "header": attrs.get('header', ''),
                "effect": attrs.get('effect', ''),
                "severity": attrs.get('severity', 0)
            })
        return alerts
    except Exception as e:
        print(f"Alert error: {e}")
        return []


def get_trip_directions_for_chat(origin, destination):
    """
    Get trip directions for chat by calling the same get_google_directions function
    that the Trip Planner uses. This ensures IDENTICAL results.
    """
    try:
        # Helper function to normalize names for matching (handle abbreviations)
        def normalize_for_matching(name):
            if not name:
                return ""
            name = name.lower().strip()
            # Common abbreviations
            name = name.replace(" rd", " road").replace(" st", " street").replace(" ave", " avenue")
            name = name.replace(" sq", " square").replace(" ctr", " center").replace(" cir", " circle")
            name = name.replace(" jct", " junction").replace(" hts", " heights")
            name = name.replace(".", "").replace(",", "").replace("'", "")
            return name
        
        # Normalize input
        origin_normalized = normalize_for_matching(origin)
        dest_normalized = normalize_for_matching(destination)
        
        print(f"Chat Trip: Looking for origin='{origin_normalized}', dest='{dest_normalized}'")
        
        # Find origin station in STATION_DATA
        origin_data = None
        origin_name = origin
        for name, data in STATION_DATA.items():
            name_normalized = normalize_for_matching(name)
            if name_normalized == origin_normalized or origin_normalized in name_normalized or name_normalized in origin_normalized:
                origin_data = data
                origin_name = name
                break
        
        # Find destination station in STATION_DATA
        dest_data = None
        dest_name = destination
        for name, data in STATION_DATA.items():
            name_normalized = normalize_for_matching(name)
            if name_normalized == dest_normalized or dest_normalized in name_normalized or name_normalized in dest_normalized:
                dest_data = data
                dest_name = name
                break
        
        # Get coordinates
        if origin_data:
            origin_coords = f"{origin_data['lat']},{origin_data['lon']}"
        else:
            origin_coords = f"{origin}, Boston, MA"
            print(f"Origin station not found in STATION_DATA, using address: {origin_coords}")
        
        if dest_data:
            dest_coords = f"{dest_data['lat']},{dest_data['lon']}"
        else:
            dest_coords = f"{destination}, Boston, MA"
            print(f"Destination station not found in STATION_DATA, using address: {dest_coords}")
        
        print(f"Chat Trip: {origin_name} ({origin_coords}) -> {dest_name} ({dest_coords})")
        
        # Call the SAME function that Trip Planner uses
        routes = get_google_directions(origin_coords, dest_coords, origin_name, dest_name)
        
        if not routes:
            print("No routes returned from get_google_directions")
            return None
        
        print(f"Chat Trip: Found {len(routes)} routes")
        
        # Return routes in a format suitable for the chat AI
        return routes
        
    except Exception as e:
        print(f"Trip directions error: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_chat_message(user_msg):
    """Main chat processing function"""
    global conversation_history, user_location
    
    intent = get_smart_intent(user_msg)
    
    if not intent:
        intent = {"intent": "general"}
    
    intent_type = intent.get('intent', 'general')
    
    data_context = {
        "intent_detected": intent_type,
        "user_query": user_msg,
        "user_location": user_location.get("station"),
        "data": {}
    }

    if intent_type in ['greeting', 'thanks', 'help', 'general']:
        if user_location.get("station"):
            data_context["data"]["user_near_station"] = user_location["station"]
            data_context["data"]["available_lines"] = user_location.get("lines", [])
    
    elif intent_type == 'alerts':
        route = intent.get('specific_route')
        route_id = resolve_route(route) if route else None
        alerts = get_alerts_for_route(route_id, limit=5)
        
        data_context["data"] = {
            "route_checked": route if route else "All System",
            "active_alerts": alerts if alerts else "None - Good Service"
        }
    
    elif intent_type == 'bus':
        bus_route = intent.get('bus_route')
        origin = intent.get('origin')
        dest = intent.get('destination')
        
        if origin == "USE_CURRENT_LOCATION" and user_location.get("station"):
            origin = user_location["station"]
            data_context["data"]["location_auto_detected"] = True
        
        if bus_route:
            predictions, stop_name = get_bus_predictions(bus_route, origin, dest)
            data_context["data"] = {
                "bus_route": bus_route,
                "stop_found": stop_name,
                "destination_filter": dest,
                "predictions": predictions
            }
        else:
            data_context["data"] = {"error": "Missing bus route number"}
    
    elif intent_type == 'commuter_rail':
        line = intent.get('commuter_line', '')
        origin = intent.get('origin')
        dest = intent.get('destination')
        
        if origin == "USE_CURRENT_LOCATION" and user_location.get("station"):
            origin = user_location["station"]
            data_context["data"]["location_auto_detected"] = True
            
        predictions, stop_name = get_commuter_rail_predictions(line, origin, dest)
        
        data_context["data"] = {
            "line": line,
            "stop_found": stop_name,
            "destination_filter": dest,
            "predictions": predictions
        }
    
    elif intent_type == 'trip_planning':
        # Handle trip planning using Google Maps Directions API
        origin = intent.get('origin')
        dest = intent.get('destination')
        
        if origin == "USE_CURRENT_LOCATION" and user_location.get("station"):
            origin = user_location["station"]
            data_context["data"]["location_auto_detected"] = True
            data_context["data"]["detected_station"] = origin
        elif origin == "USE_CURRENT_LOCATION" and user_location.get("lat"):
            # Use coordinates directly
            origin_coords = f"{user_location['lat']},{user_location['lon']}"
            data_context["data"]["location_auto_detected"] = True
            data_context["data"]["using_gps_coords"] = True
        
        if not origin:
            data_context["data"] = {
                "error": "Missing Origin",
                "message": "I need to know where you're starting from. Please tell me your starting point or click the location button.",
                "request_location": True
            }
        elif not dest:
            data_context["data"] = {
                "error": "Missing Destination",
                "message": "Where would you like to go? Please tell me your destination."
            }
        else:
            # Get directions from Google Maps API
            routes = get_trip_directions_for_chat(origin, dest)
            
            if routes:
                data_context["data"] = {
                    "origin": origin,
                    "destination": dest,
                    "routes": routes,
                    "num_routes": len(routes),
                    "location_auto_detected": data_context.get("data", {}).get("location_auto_detected", False)
                }
            else:
                data_context["data"] = {
                    "error": "No Routes Found",
                    "origin": origin,
                    "destination": dest,
                    "message": "I couldn't find transit routes between these locations. Please check the station names or try the Trip Planner tool."
                }
    
    elif intent_type == 'prediction':
        origin = intent.get('origin')
        dest = intent.get('destination')
        offset = intent.get('time_offset', 0)
        specific_route = intent.get('specific_route')
        
        if origin == "USE_CURRENT_LOCATION":
            if user_location.get("station"):
                origin = user_location["station"]
                data_context["data"]["location_auto_detected"] = True
                data_context["data"]["detected_station"] = origin
            else:
                data_context["data"] = {
                    "error": "Location Required",
                    "message": "I need your location to find nearby trains. Please click the location button.",
                    "request_location": True
                }
                final_reply = generate_natural_response(user_msg, data_context)
                conversation_history.append({"role": "assistant", "content": final_reply})
                return {"reply": final_reply, "request_location": True}
        
        if origin: origin = normalize_station_name(origin)
        if dest: dest = normalize_station_name(dest)
        
        stop_id = STATION_TO_ID.get(origin) if origin else None
        
        if origin and dest and not specific_route:
            deduced_route = find_common_route(origin, dest)
            if deduced_route:
                specific_route = deduced_route
                data_context["data"]["auto_deduced_route"] = deduced_route

        if not stop_id:
            if user_location.get("station"):
                data_context["data"] = {
                    "error": "Unknown Origin", 
                    "suggestion": f"Did you mean to start from {user_location['station']}?"
                }
            else:
                data_context["data"] = {"error": "Unknown Origin", "details": f"Could not find station '{origin}'"}
        elif not dest:
            data_context["data"] = {"error": "Missing Destination", "origin_locked": origin}
        else:
            dest = normalize_station_name(dest)
            route_data = DEFAULT_ROUTES.get(dest)
            
            if specific_route:
                if route_data:
                    route_data = {"route": specific_route, "dir": route_data['dir']}
                else:
                    route_data = {"route": specific_route, "dir": 1}
            
            if not route_data:
                 data_context["data"] = {
                     "error": "Route Ambiguous", 
                     "origin": origin, 
                     "dest": dest,
                     "message": "I need to know the Line (Red, Orange, Blue, Green) to check."
                 }
            else:
                url = f"{BASE_URL}/predictions"
                params = {
                    "filter[stop]": stop_id,
                    "filter[route]": route_data['route'],
                    "filter[direction_id]": route_data['dir'],
                    "include": "schedule",
                    "sort": "arrival_time",
                    "page[limit]": 8
                }

                try:
                    response = requests.get(url, headers=HEADERS, params=params)
                    api_data = response.json()
                    
                    trains_found = []
                    now = datetime.now()
                    
                    for train in api_data.get('data', []):
                        arrival = train['attributes']['arrival_time'] or train['attributes']['departure_time']
                        if not arrival: continue
                        
                        arrival_dt = parser.isoparse(arrival)
                        if arrival_dt.tzinfo is None:
                            arrival_dt = arrival_dt.replace(tzinfo=now.astimezone().tzinfo)
                        else:
                            now = now.astimezone(arrival_dt.tzinfo)
                        
                        diff_min = int((arrival_dt - now).total_seconds() / 60)
                        
                        if diff_min >= offset:
                            trains_found.append({
                                "minutes_away": diff_min,
                                "arrival_time": format_time(arrival),
                                "status": train['attributes'].get('status', 'On Time')
                            })
                    
                    data_context["data"] = {
                        "route": route_data['route'],
                        "origin": origin,
                        "destination": dest,
                        "predictions": trains_found,
                        "typo_correction": intent.get('original_station_mentioned'),
                        "location_auto_detected": data_context.get("data", {}).get("location_auto_detected", False)
                    }

                except Exception as e:
                    print(f"MBTA API Error: {e}")
                    data_context["data"] = {"error": "API Connection Failed"}

    final_reply = generate_natural_response(user_msg, data_context)
    conversation_history.append({"role": "assistant", "content": final_reply})
    
    return {"reply": final_reply}


# ==================== FLASK ROUTES ====================

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "service": "charlie-mbta-chatbot",
        "version": "2.1.0",
        "features": ["chatbot", "location", "map", "bold-timing"]
    })


# ==================== LOCATION API ENDPOINTS ====================

@app.route('/api/set-location', methods=['POST'])
def set_location():
    global user_location
    
    data = request.get_json()
    lat = data.get('lat')
    lon = data.get('lon')
    
    if not lat or not lon:
        return jsonify({"error": "Missing lat/lon"}), 400
    
    nearest = find_nearest_station(lat, lon)
    
    if nearest:
        user_location = {
            "lat": lat,
            "lon": lon,
            "station": nearest["name"],
            "stop_id": nearest["id"],
            "distance": nearest["distance"],
            "lines": nearest["lines"]
        }
        return jsonify({
            "success": True,
            "nearest_station": nearest,
            "message": f"📍 You're near {nearest['name']} ({nearest['distance']}m away)"
        })
    else:
        user_location = {"lat": lat, "lon": lon, "station": None, "stop_id": None, "lines": []}
        return jsonify({
            "success": False,
            "message": "No MBTA stations found within 2km of your location."
        })


@app.route('/api/get-location')
def get_location():
    return jsonify(user_location)


@app.route('/api/stations')
def get_all_stations():
    stations_list = []
    seen_ids = set()
    
    for name, data in STATION_DATA.items():
        if data["id"] in seen_ids:
            continue
        seen_ids.add(data["id"])
        
        stations_list.append({
            "name": name,
            "id": data["id"],
            "lat": data["lat"],
            "lon": data["lon"],
            "lines": data["lines"]
        })
    
    return jsonify(stations_list)


@app.route('/api/nearest-station')
def nearest_station():
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    
    if not lat or not lon:
        return jsonify({"error": "Missing lat/lon parameters"}), 400
    
    nearest = find_nearest_station(lat, lon)
    
    if nearest:
        return jsonify(nearest)
    else:
        return jsonify({"error": "No stations found nearby"}), 404


# ==================== EXISTING API ENDPOINTS ====================

@app.route('/api/chat', methods=['POST'])
def chat_post():
    data = request.get_json()
    user_msg = data.get('message', '').strip()
    if not user_msg:
        return jsonify({"reply": "👀 Type something!"})
    return jsonify(process_chat_message(user_msg))


@app.route('/chat')
def chat_get():
    user_msg = request.args.get('msg', '').strip()
    if not user_msg:
        return jsonify({"reply": "👀 Type something!"})
    return jsonify(process_chat_message(user_msg))


@app.route('/api/routes')
def get_routes():
    try:
        response = requests.get(f"{BASE_URL}/routes", headers=HEADERS)
        data = response.json()
        routes_by_type = {"subway": [], "bus": [], "commuter_rail": [], "ferry": []}
        
        for route in data.get('data', []):
            route_id = route['id']
            attrs = route['attributes']
            route_type = attrs.get('type', 3)
            
            route_info = {
                "id": route_id,
                "name": attrs.get('long_name') or attrs.get('short_name', route_id),
                "short_name": attrs.get('short_name', ''),
                "color": f"#{attrs.get('color', '000000')}",
                "text_color": f"#{attrs.get('text_color', 'FFFFFF')}",
                "direction_names": attrs.get('direction_names', ['Outbound', 'Inbound']),
                "direction_destinations": attrs.get('direction_destinations', ['', ''])
            }
            
            if route_type in [0, 1]: routes_by_type["subway"].append(route_info)
            elif route_type == 2: routes_by_type["commuter_rail"].append(route_info)
            elif route_type == 3: routes_by_type["bus"].append(route_info)
            elif route_type == 4: routes_by_type["ferry"].append(route_info)
        
        return jsonify(routes_by_type)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/alerts')
def get_alerts():
    try:
        route_filter = request.args.get('route')
        params = {"sort": "-severity,-updated_at", "page[limit]": 100}
        if route_filter:
            params["filter[route]"] = route_filter
        
        response = requests.get(f"{BASE_URL}/alerts", headers=HEADERS, params=params)
        data = response.json()
        
        alerts = []
        for alert in data.get('data', []):
            attrs = alert['attributes']
            affected_routes = []
            for entity in attrs.get('informed_entity', []):
                if 'route' in entity:
                    affected_routes.append(entity['route'])
            
            alerts.append({
                "id": alert['id'],
                "effect": attrs.get('effect', 'UNKNOWN'),
                "severity": attrs.get('severity', 0),
                "header": attrs.get('header', ''),
                "description": attrs.get('description', ''),
                "affected_routes": list(set(affected_routes)),
                "updated_at": attrs.get('updated_at')
            })
        
        return jsonify({"alerts": alerts, "count": len(alerts)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/predictions')
def get_predictions():
    try:
        stop_id = request.args.get('stop')
        route_id = request.args.get('route')
        direction_id = request.args.get('direction')
        
        if not stop_id and not route_id:
            return jsonify({"error": "Provide stop or route"}), 400
        
        params = {"include": "stop,route,trip,vehicle", "sort": "arrival_time", "page[limit]": 20}
        if stop_id: params["filter[stop]"] = stop_id
        if route_id: params["filter[route]"] = route_id
        if direction_id: params["filter[direction_id]"] = direction_id
        
        response = requests.get(f"{BASE_URL}/predictions", headers=HEADERS, params=params)
        data = response.json()
        
        included = {f"{item['type']}_{item['id']}": item for item in data.get('included', [])}
        
        predictions = []
        for pred in data.get('data', []):
            attrs = pred['attributes']
            rels = pred.get('relationships', {})
            
            route_data = rels.get('route', {}).get('data', {})
            stop_data = rels.get('stop', {}).get('data', {})
            trip_data = rels.get('trip', {}).get('data', {})
            
            route_info = included.get(f"route_{route_data.get('id', '')}", {})
            stop_info = included.get(f"stop_{stop_data.get('id', '')}", {})
            trip_info = included.get(f"trip_{trip_data.get('id', '')}", {})
            
            arrival_time = attrs.get('arrival_time') or attrs.get('departure_time')
            if not arrival_time: continue
            
            arrival_dt = parser.isoparse(arrival_time)
            route_attrs = route_info.get('attributes', {})
            
            predictions.append({
                "id": pred['id'],
                "arrival_time": arrival_time,
                "arrival_formatted": format_time(arrival_time),
                "time_away": format_time_diff(arrival_dt),
                "status": attrs.get('status', ''),
                "direction_id": attrs.get('direction_id'),
                "route": {
                    "id": route_data.get('id', ''),
                    "name": route_attrs.get('long_name') or route_attrs.get('short_name', ''),
                    "color": f"#{route_attrs.get('color', '000000')}"
                },
                "stop": {
                    "id": stop_data.get('id', ''),
                    "name": stop_info.get('attributes', {}).get('name', '')
                },
                "trip": {
                    "id": trip_data.get('id', ''),
                    "headsign": trip_info.get('attributes', {}).get('headsign', '')
                }
            })
        
        return jsonify({"predictions": predictions, "count": len(predictions)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/vehicles')
def get_vehicles():
    try:
        route_id = request.args.get('route')
        params = {"include": "stop,route,trip", "page[limit]": 100}
        if route_id: params["filter[route]"] = route_id
        
        response = requests.get(f"{BASE_URL}/vehicles", headers=HEADERS, params=params)
        data = response.json()
        
        included = {f"{item['type']}_{item['id']}": item for item in data.get('included', [])}
        
        vehicles = []
        for vehicle in data.get('data', []):
            attrs = vehicle['attributes']
            rels = vehicle.get('relationships', {})
            
            route_data = rels.get('route', {}).get('data', {})
            route_info = included.get(f"route_{route_data.get('id', '')}", {})
            route_attrs = route_info.get('attributes', {})
            
            vehicles.append({
                "id": vehicle['id'],
                "label": attrs.get('label', ''),
                "latitude": attrs.get('latitude'),
                "longitude": attrs.get('longitude'),
                "bearing": attrs.get('bearing'),
                "current_status": attrs.get('current_status', ''),
                "direction_id": attrs.get('direction_id'),
                "route": {
                    "id": route_data.get('id', ''),
                    "name": route_attrs.get('long_name') or route_attrs.get('short_name', ''),
                    "color": f"#{route_attrs.get('color', '000000')}"
                }
            })
        
        return jsonify({"vehicles": vehicles, "count": len(vehicles)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/stops')
def get_stops():
    try:
        route_id = request.args.get('route')
        params = {"page[limit]": 100}
        if route_id: params["filter[route]"] = route_id
        
        response = requests.get(f"{BASE_URL}/stops", headers=HEADERS, params=params)
        data = response.json()
        
        stops = []
        for stop in data.get('data', []):
            attrs = stop['attributes']
            stops.append({
                "id": stop['id'],
                "name": attrs.get('name', ''),
                "latitude": attrs.get('latitude'),
                "longitude": attrs.get('longitude'),
                "municipality": attrs.get('municipality', ''),
                "wheelchair_boarding": attrs.get('wheelchair_boarding')
            })
        
        return jsonify({"stops": stops, "count": len(stops)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/line/<line_id>')
def get_line_status(line_id):
    try:
        pred_response = requests.get(f"{BASE_URL}/predictions", headers=HEADERS,
            params={"filter[route]": line_id, "include": "stop,trip", "sort": "arrival_time", "page[limit]": 50})
        pred_data = pred_response.json()
        
        vehicle_response = requests.get(f"{BASE_URL}/vehicles", headers=HEADERS,
            params={"filter[route]": line_id})
        vehicle_data = vehicle_response.json()
        
        alert_response = requests.get(f"{BASE_URL}/alerts", headers=HEADERS,
            params={"filter[route]": line_id, "sort": "-severity"})
        alert_data = alert_response.json()
        
        included = {f"{item['type']}_{item['id']}": item for item in pred_data.get('included', [])}
        
        predictions_by_stop = {}
        for pred in pred_data.get('data', []):
            attrs = pred['attributes']
            rels = pred.get('relationships', {})
            stop_data = rels.get('stop', {}).get('data', {})
            trip_data = rels.get('trip', {}).get('data', {})
            stop_info = included.get(f"stop_{stop_data.get('id', '')}", {})
            trip_info = included.get(f"trip_{trip_data.get('id', '')}", {})
            
            stop_id = stop_data.get('id', '')
            stop_name = stop_info.get('attributes', {}).get('name', stop_id)
            arrival_time = attrs.get('arrival_time') or attrs.get('departure_time')
            if not arrival_time: continue
            
            arrival_dt = parser.isoparse(arrival_time)
            if stop_id not in predictions_by_stop:
                predictions_by_stop[stop_id] = {"stop_id": stop_id, "stop_name": stop_name, "predictions": []}
            
            predictions_by_stop[stop_id]["predictions"].append({
                "time": arrival_time,
                "time_formatted": format_time(arrival_time),
                "time_away": format_time_diff(arrival_dt),
                "headsign": trip_info.get('attributes', {}).get('headsign', ''),
                "direction_id": attrs.get('direction_id')
            })
        
        vehicles = [{"id": v['id'], "label": v['attributes'].get('label', ''),
                    "latitude": v['attributes'].get('latitude'), "longitude": v['attributes'].get('longitude'),
                    "current_status": v['attributes'].get('current_status', '')}
                   for v in vehicle_data.get('data', [])]
        
        alerts = [{"id": a['id'], "effect": a['attributes'].get('effect', ''),
                  "severity": a['attributes'].get('severity', 0), "header": a['attributes'].get('header', '')}
                 for a in alert_data.get('data', [])]
        
        route_response = requests.get(f"{BASE_URL}/routes/{line_id}", headers=HEADERS)
        route_attrs = route_response.json().get('data', {}).get('attributes', {})
        
        return jsonify({
            "line": {
                "id": line_id,
                "name": route_attrs.get('long_name') or route_attrs.get('short_name', line_id),
                "color": f"#{route_attrs.get('color', '000000')}"
            },
            "predictions_by_stop": list(predictions_by_stop.values()),
            "vehicles": vehicles,
            "alerts": alerts
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/dashboard')
def get_dashboard():
    try:
        alert_response = requests.get(f"{BASE_URL}/alerts", headers=HEADERS,
            params={"sort": "-severity", "page[limit]": 50})
        alert_data = alert_response.json()
        
        alerts_by_route = {}
        severe_alerts = []
        
        for alert in alert_data.get('data', []):
            attrs = alert['attributes']
            alert_info = {
                "id": alert['id'],
                "effect": attrs.get('effect', ''),
                "severity": attrs.get('severity', 0),
                "header": attrs.get('header', ''),
                "short_header": attrs.get('short_header', '')
            }
            if attrs.get('severity', 0) >= 7:
                severe_alerts.append(alert_info)
            
            for entity in attrs.get('informed_entity', []):
                route = entity.get('route')
                if route:
                    if route not in alerts_by_route: alerts_by_route[route] = []
                    alerts_by_route[route].append(alert_info)
        
        subway_status = {}
        for line in SUBWAY_ROUTES:
            vehicle_response = requests.get(f"{BASE_URL}/vehicles", headers=HEADERS,
                params={"filter[route]": line})
            vehicle_count = len(vehicle_response.json().get('data', []))
            line_alerts = alerts_by_route.get(line, [])
            has_severe = any(a['severity'] >= 7 for a in line_alerts)
            
            subway_status[line] = {
                "name": line,
                "color": LINE_COLORS.get(line, "#000000"),
                "vehicle_count": vehicle_count,
                "alert_count": len(line_alerts),
                "status": "alert" if has_severe else ("warning" if line_alerts else "normal")
            }
        
        return jsonify({
            "subway_status": subway_status,
            "severe_alerts": severe_alerts[:10],
            "total_alerts": len(alert_data.get('data', [])),
            "updated_at": get_now().isoformat(),
            "user_location": user_location
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/station/<stop_id>')
def get_station_info(stop_id):
    try:
        stop_response = requests.get(f"{BASE_URL}/stops/{stop_id}", headers=HEADERS)
        stop_attrs = stop_response.json().get('data', {}).get('attributes', {})
        
        pred_response = requests.get(f"{BASE_URL}/predictions", headers=HEADERS,
            params={"filter[stop]": stop_id, "include": "route,trip", "sort": "arrival_time", "page[limit]": 30})
        pred_data = pred_response.json()
        
        included = {f"{item['type']}_{item['id']}": item for item in pred_data.get('included', [])}
        
        predictions = []
        for pred in pred_data.get('data', []):
            attrs = pred['attributes']
            rels = pred.get('relationships', {})
            route_data = rels.get('route', {}).get('data', {})
            trip_data = rels.get('trip', {}).get('data', {})
            route_info = included.get(f"route_{route_data.get('id', '')}", {})
            trip_info = included.get(f"trip_{trip_data.get('id', '')}", {})
            
            arrival_time = attrs.get('arrival_time') or attrs.get('departure_time')
            if not arrival_time: continue
            
            arrival_dt = parser.isoparse(arrival_time)
            route_attrs = route_info.get('attributes', {})
            
            predictions.append({
                "route_id": route_data.get('id', ''),
                "route_name": route_attrs.get('long_name') or route_attrs.get('short_name', ''),
                "route_color": f"#{route_attrs.get('color', '000000')}",
                "headsign": trip_info.get('attributes', {}).get('headsign', ''),
                "arrival_formatted": format_time(arrival_time),
                "time_away": format_time_diff(arrival_dt)
            })
        
        return jsonify({
            "station": {
                "id": stop_id,
                "name": stop_attrs.get('name', ''),
                "latitude": stop_attrs.get('latitude'),
                "longitude": stop_attrs.get('longitude')
            },
            "predictions": predictions
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== GOOGLE MAPS TRIP PLANNER API ====================
# Uses Google Maps Directions API for accurate multi-modal routing

GOOGLE_MAPS_API_KEY = os.environ.get('GOOGLE_MAPS_API_KEY', 'AIzaSyBZpsvc9e-tQE_i3VtKNUTP6aEnsKcCPK8')


@app.route('/api/trip-planner', methods=['POST'])
def plan_trip():
    """
    Plan a trip using Google Maps Directions API.
    Returns ALL possible transit routes like Google Maps.
    """
    data = request.get_json()
    origin = data.get('origin', '')
    destination = data.get('destination', '')
    origin_coords = data.get('origin_coords')
    dest_coords = data.get('dest_coords')
    
    if not origin or not destination:
        return jsonify({"error": "Origin and destination are required"}), 400
    
    # Get coordinates if not provided
    if not origin_coords:
        origin_data = find_station_data(origin)
        if origin_data:
            origin_coords = f"{origin_data['lat']},{origin_data['lon']}"
        else:
            origin_coords = f"{origin}, Boston, MA"
    
    if not dest_coords:
        dest_data = find_station_data(destination)
        if dest_data:
            dest_coords = f"{dest_data['lat']},{dest_data['lon']}"
        else:
            dest_coords = f"{destination}, Boston, MA"
    
    # Call Google Maps Directions API
    routes = get_google_directions(origin_coords, dest_coords, origin, destination)
    
    return jsonify({
        "origin": origin,
        "destination": destination,
        "routes": routes
    })


def get_google_directions(origin, destination, origin_name, dest_name):
    """
    Get transit directions from Google Maps Directions API.
    Returns multiple route alternatives with all transit modes.
    """
    try:
        url = "https://maps.googleapis.com/maps/api/directions/json"
        params = {
            "origin": origin,
            "destination": destination,
            "mode": "transit",
            "alternatives": "true",
            "transit_mode": "bus|subway|train|tram|rail",
            "key": GOOGLE_MAPS_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
        
        if data.get('status') != 'OK':
            print(f"Google Maps API error: {data.get('status')} - {data.get('error_message', '')}")
            return []
        
        routes = []
        
        for route in data.get('routes', []):
            leg = route.get('legs', [{}])[0]
            
            duration_text = leg.get('duration', {}).get('text', '')
            duration_value = leg.get('duration', {}).get('value', 0) // 60  # Convert to minutes
            
            depart_time = leg.get('departure_time', {}).get('text', '')
            arrive_time = leg.get('arrival_time', {}).get('text', '')
            
            # Process steps
            steps = []
            detailed_steps = []
            transfers = -1  # Will count transit steps - 1
            walk_time = 0
            fare = None
            
            for step in leg.get('steps', []):
                travel_mode = step.get('travel_mode', '')
                duration = step.get('duration', {}).get('text', '')
                instruction = step.get('html_instructions', '').replace('<b>', '').replace('</b>', '').replace('<div style="font-size:0.9em">', ' - ').replace('</div>', '')
                
                if travel_mode == 'WALKING':
                    walk_duration = step.get('duration', {}).get('value', 0) // 60
                    walk_time += walk_duration
                    
                    detailed_steps.append({
                        "type": "walk",
                        "instruction": f"Walk {step.get('distance', {}).get('text', '')}",
                        "details": instruction,
                        "duration": duration
                    })
                    
                elif travel_mode == 'TRANSIT':
                    transfers += 1
                    transit_details = step.get('transit_details', {})
                    line = transit_details.get('line', {})
                    
                    # Get transit info
                    vehicle_type = line.get('vehicle', {}).get('type', 'BUS')
                    line_name = line.get('short_name') or line.get('name', '')
                    line_color = line.get('color', '#666666')
                    headsign = transit_details.get('headsign', '')
                    
                    departure_stop = transit_details.get('departure_stop', {}).get('name', '')
                    arrival_stop = transit_details.get('arrival_stop', {}).get('name', '')
                    departure_time_step = transit_details.get('departure_time', {}).get('text', '')
                    arrival_time_step = transit_details.get('arrival_time', {}).get('text', '')
                    num_stops = transit_details.get('num_stops', 0)
                    
                    # Determine step type
                    step_type = get_transit_type(vehicle_type, line_name)
                    
                    # Add to visual steps (route summary)
                    steps.append({
                        "type": step_type,
                        "name": get_display_name(line_name, vehicle_type),
                        "short_name": line_name or vehicle_type[:3],
                        "color": line_color,
                        "vehicle_type": vehicle_type
                    })
                    
                    # Add detailed steps
                    detailed_steps.append({
                        "type": "transit",
                        "instruction": f"Take {get_display_name(line_name, vehicle_type)}",
                        "details": f"Board at {departure_stop}" + (f" toward {headsign}" if headsign else ""),
                        "time": departure_time_step,
                        "duration": duration,
                        "color": line_color,
                        "num_stops": num_stops
                    })
                    
                    detailed_steps.append({
                        "type": "transit",
                        "instruction": f"Get off at {arrival_stop}",
                        "details": f"{num_stops} stop{'s' if num_stops != 1 else ''}" if num_stops else "Arrive",
                        "time": arrival_time_step,
                        "color": line_color
                    })
            
            # Get fare if available
            if 'fare' in route:
                fare = route['fare'].get('text', '')
            elif 'fare' in leg:
                fare = leg['fare'].get('text', '')
            
            # Build route object
            route_obj = {
                "duration": duration_value,
                "duration_text": duration_text,
                "transfers": max(0, transfers),
                "depart_time": depart_time,
                "arrive_time": arrive_time,
                "time_range": f"{depart_time}—{arrive_time}" if depart_time and arrive_time else duration_text,
                "walk_time": walk_time,
                "fare": fare or estimate_fare(steps),
                "steps": steps,
                "detailed_steps": detailed_steps,
                "warnings": route.get('warnings', []),
                "summary": route.get('summary', '')
            }
            
            routes.append(route_obj)
        
        # Sort by duration
        routes.sort(key=lambda x: x['duration'])
        
        return routes[:6]  # Return top 6 routes
        
    except Exception as e:
        print(f"Google Directions API error: {e}")
        import traceback
        traceback.print_exc()
        return []


def get_transit_type(vehicle_type, line_name):
    """Convert Google's vehicle type to our display type"""
    vehicle_type = vehicle_type.upper()
    line_name = (line_name or '').upper()
    
    # Check for specific MBTA lines
    if 'RED' in line_name or 'ORANGE' in line_name or 'BLUE' in line_name or 'GREEN' in line_name:
        return 'subway'
    
    if 'SL' in line_name or 'SILVER' in line_name:
        return 'silver-line'
    
    if vehicle_type in ['SUBWAY', 'METRO_RAIL', 'HEAVY_RAIL']:
        return 'subway'
    elif vehicle_type in ['TRAM', 'LIGHT_RAIL']:
        return 'tram'
    elif vehicle_type in ['COMMUTER_TRAIN', 'RAIL', 'LONG_DISTANCE_TRAIN']:
        return 'commuter-rail'
    elif vehicle_type == 'BUS':
        return 'bus'
    elif vehicle_type == 'FERRY':
        return 'ferry'
    else:
        return 'transit'


def get_display_name(line_name, vehicle_type):
    """Get a nice display name for the transit line"""
    if not line_name:
        return vehicle_type.replace('_', ' ').title()
    
    # Handle MBTA specific names
    line_upper = line_name.upper()
    
    if line_upper in ['RED', 'ORANGE', 'BLUE']:
        return f"{line_name} Line"
    elif 'GREEN' in line_upper:
        return f"Green Line {line_name.replace('Green-', '').replace('Green ', '')}" if '-' in line_name or ' ' in line_name else "Green Line"
    elif line_upper.startswith('SL') or 'SILVER' in line_upper:
        return f"Silver Line {line_name.replace('SL', '').replace('Silver', '').strip()}"
    elif line_upper.startswith('CR-'):
        return f"{line_name.replace('CR-', '')} Line (CR)"
    elif vehicle_type in ['BUS']:
        return f"Bus {line_name}"
    elif vehicle_type in ['COMMUTER_TRAIN', 'RAIL']:
        return f"{line_name} Line"
    else:
        return line_name


def estimate_fare(steps):
    """Estimate fare based on transit types used"""
    if not steps:
        return "$2.40"
    
    has_subway = any(s.get('type') == 'subway' for s in steps)
    has_bus = any(s.get('type') == 'bus' for s in steps)
    has_cr = any(s.get('type') == 'commuter-rail' for s in steps)
    has_ferry = any(s.get('type') == 'ferry' for s in steps)
    
    if has_cr:
        return "$6.50+"  # CR fares vary by zone
    if has_ferry:
        return "$3.70"
    if has_subway and has_bus:
        return "$2.40"  # Free transfer from subway to bus
    if has_subway:
        return "$2.40"
    if has_bus:
        return "$1.70"
    
    return "$2.40"


def find_station_data(query):
    """Find station data by name"""
    if not query:
        return None
    
    query_lower = query.lower().strip()
    
    # Try exact match first
    for name, data in STATION_DATA.items():
        if name.lower() == query_lower:
            return {"name": name, **data}
    
    # Try partial match
    for name, data in STATION_DATA.items():
        if query_lower in name.lower():
            return {"name": name, **data}
    
    return None


if __name__ == '__main__':
    import sys
    port = int(os.environ.get('PORT', 5001))
    if '--port' in sys.argv:
        idx = sys.argv.index('--port') + 1
        if idx < len(sys.argv): port = int(sys.argv[idx])
    
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║  🚇 MBTA Live Transit Dashboard with AI Chatbot + Location    ║
║  ─────────────────────────────────────────────────────────────║
║  Server: http://localhost:{port}                               ║
║  MBTA API: {'✅ Connected' if MBTA_API_KEY else '❌ Missing'}                                    ║
║  OpenAI:   {'✅ Connected' if OPENAI_API_KEY else '❌ Missing (basic chatbot)'}                  ║
║  Google:   {'✅ Connected' if GOOGLE_MAPS_API_KEY else '❌ Missing'}                             ║
║  Features: Chat, Map, Location, Trip Planner                  ║
╚═══════════════════════════════════════════════════════════════╝
""")
    
    app.run(debug=True, host='0.0.0.0', port=port)
