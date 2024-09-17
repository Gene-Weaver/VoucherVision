import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(layout="wide", page_title='Text Difference Tool with Quick Diff',initial_sidebar_state="collapsed")

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
if st.button("Compare Texts"):
    if text1 and text2:
        # Embedding the JavaScript Quick Diff via an iframe
        quick_diff_html = f"""
        <iframe 
            src="http://localhost:3000/?left={text1}&right={text2}" 
            width="100%" 
            height="600px"
        ></iframe>
        """
        components.html(quick_diff_html, height=600)
    else:
        st.warning("Please provide text in both fields.")
