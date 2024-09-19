import streamlit as st
import streamlit.components.v1 as components
import json, time  
from urllib.parse import quote 

st.set_page_config(layout="wide", page_title='Text Difference Tool with Quick Diff',initial_sidebar_state="collapsed")

if 'frame_height' not in st.session_state:
    st.session_state.frame_height = 600 
if 'favorite' not in st.session_state:
    st.session_state['favorite'] = None  # Initial state for favorite text

def set_favorite(selected_text):
    st.session_state['favorite'] = selected_text


# Streamlit App Code
st.title("Text Difference Tool with Quick Diff")

# Text areas for user input
text1 = {
    "filename": "MICH_16205594_Poaceae_Jouvea_pilosa",
    "catalogNumber": "1122841",
    "order": "Poales",
    "family": "Poaceae",
    "scientificName": "Distichlis spicata (L.) Greene",
    "genus": "Distichlis",
    "specificEpithet": "spicata",
    "speciesNameAuthorship": "(L.) Greene",
    "collectedBy": "",
    "collectorNumber": "",
    "identifiedBy": "",
    "verbatimCollectionDate": "29 April 1973",
    "collectionDate": "1973-04-29",
    "collectionDateEnd": "",
    "specimenNotes": "Volcanic islet--cormorant and tern breeding ground. In open, level to slightly sloping areas favored by the terns for nesting, near the beach; surrounded by low forest consisting almost entirely of Crataeva tapia (3-4 m tall), Porous soil.",
    "habitat": "Volcanic islet--cormorant and tern breeding ground. In open, level to slightly sloping areas favored by the terns for nesting, near the beach; surrounded by low forest consisting almost entirely of Crataeva tapia (3-4 m tall), Porous soil.",
    "cultivated": "",
    "country": "Mexico",
    "stateProvince": "Nayarit",
    "county": "",
    "locality": "Isabel Island, E of Tres Marias Is.",
    "verbatimCoordinates": "",
    "decimalLatitude": "",
    "decimalLongitude": "",
    "minimumElevationInMeters": "",
    "maximumElevationInMeters": "",
    "elevationUnits": ""
}
text2 = {
    "filename": "MICH_16205594_Poaceae_Jouvea_pilosa",
    "catalogNumber": "1122841",
    "order": "Poales",
    "family": "Poaceae",
    "scientificName": "Distichlis spicata (L.) Greene",
    "genus": "Distichlis",
    "specificEpithet": "spicata",
    "speciesNameAuthorship": "(L) Greene",
    "collectedBy": "",
    "collectorNumber": "",
    "identifiedBy": "",
    "verbatimCollectionDate": "29 April 1973",
    "collectionDate": "1973-04-29",
    "collectionDateEnd": "",
    "specimenNotes": "Volcanic islet--cormorant and tern breeding ground. In open, level to slightly sloping areas favored by the terns for nesting, near the beach; surrounded by low forest consisting almost entirely of Crataeva tapia (3-4 m tall), Porous soil.",
    "habitat": "Volcanic islet--cormorant and tern breeding ground. In open, level to slightly sloping areas favored by the terns for nesting, near the beach; surrounded by low forest consisting almost entirely of Crataeva tapia (3-4 m tall), Porous soil.",
    "cultivated": "",
    "country": "Mexico",
    "stateProvince": "",
    "county": "",
    "locality": "Isabel Island, E of Tres Marias Is.",
    "verbatimCoordinates": "",
    "decimalLatitude": "",
    "decimalLongitude": "",
    "minimumElevationInMeters": "",
    "maximumElevationInMeters": "",
    "elevationUnits": ""
}

# On button click, embed the Quick Diff tool
st.session_state.frame_height = st.sidebar.slider("Viewing Height", 400, 1100, 600, 50)

col_left, col_right = st.columns(2)
with col_left:
    left_favorite = st.button("Favorite Left")
with col_right:
    right_favorite = st.button("Favorite Right")

# Set the favorite text when a button is clicked
if left_favorite:
    set_favorite(text1)
if right_favorite:
    set_favorite(text2)

# Display the selected favorite text
if st.session_state['favorite']:
    st.write(f"Your favorite text: {st.session_state['favorite']}")

# Serialize the text1 and text2 to JSON and then URL encode them
text1_json = quote(json.dumps(text1))  # URL encode the JSON data
text2_json = quote(json.dumps(text2))  # URL encode the JSON data

# Embed the Quick Diff tool in an iframe
if text1_json and text2_json:
    # Embedding the JavaScript Quick Diff via an iframe
    timestamp = int(time.time())  # Use the current timestamp as a unique cache buster

    quick_diff_html = f"""
    <iframe 
        src="http://localhost:3000/?left={text1_json}&right={text2_json}&t={timestamp}" 
        width="100%" 
        height="1000px"
    ></iframe>
    """
    components.html(quick_diff_html, height=st.session_state.frame_height)
else:
    st.warning("Please provide text in both fields.")
