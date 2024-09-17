import streamlit as st
import streamlit.components.v1 as components

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
text1 = [
  {
    "id": "0002",
    "type": "Ice Cream",
    "name": "Cake",
    "ppu": 0.55,
    "batters": {
      "batter": [
        {
          "id": "1001",
          "type": "Regular"
        },
        {
          "id": "1002",
          "type": "Chocolate"
        },
        {
          "id": "1003",
          "type": "Blueberry"
        },
        {
          "id": "1004",
          "type": "Devil's Food"
        }
      ]
    },
    "topping": [
      {
        "id": "5001",
        "type": "None"
      },
      {
        "id": "5002",
        "type": "Glazed"
      },
      {
        "id": "5007",
        "type": "Powdered Sugar"
      },
      {
        "id": "5004",
        "type": "Maple"
      }
    ]
  }
]
text2 = [
  {
    "id": "0001",
    "type": "donut",
    "name": "Cake",
    "ppu": 0.55,
    "batters": {
      "batter": [
        {
          "id": "1002",
          "type": "Chocolate"
        }
      ]
    },
    "topping": [
      {
        "id": "5007",
        "type": "Powdered Sugar"
      },
      {
        "id": "5006",
        "type": "Chocolate with Sprinkles"
      },
      {
        "id": "5004",
        "type": "Maple"
      }
    ]
  }
]

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

# if st.button("Compare Texts"):
if text1 and text2:
    # Embedding the JavaScript Quick Diff via an iframe
    quick_diff_html = f"""
    <iframe 
        src="http://localhost:3000/?left={text1}&right={text2}" 
        width="100%" 
        height="1000px"
    ></iframe>
    """
    components.html(quick_diff_html, height=st.session_state.frame_height)
else:
    st.warning("Please provide text in both fields.")
