import os
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(layout="wide", page_icon='img/icon.ico', page_title='VV Report Bugs',initial_sidebar_state="collapsed")

def display_report():
    c1, c2, c3 = st.columns([4,6,1])
    with c3:
        try:
            st.page_link('app.py', label="Home", icon="🏠")
            st.page_link(os.path.join("pages","faqs.py"), label="FAQs", icon="❔")
            st.page_link(os.path.join("pages","report_bugs.py"), label="Report a Bug", icon="⚠️")
        except:
            st.page_link(os.path.join(os.path.dirname(os.path.dirname(__file__)),'app.py'), label="Home", icon="🏠")
            st.page_link(os.path.join(os.path.dirname(os.path.dirname(__file__)),"pages","faqs.py"), label="FAQs", icon="❔")
            st.page_link(os.path.join(os.path.dirname(os.path.dirname(__file__)),"pages","report_bugs.py"), label="Report a Bug", icon="⚠️")

    with c2:
        st.write('To report a bug or request a new feature please fill out this [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSdtW1z9Q1pGZTo5W9UeCa6PlQanP-b88iNKE6zsusRI78Itsw/viewform?usp=sf_link)')
        components.iframe(f"https://docs.google.com/forms/d/e/1FAIpQLSdtW1z9Q1pGZTo5W9UeCa6PlQanP-b88iNKE6zsusRI78Itsw/viewform?embedded=true", height=700,scrolling=True,width=640)


display_report()