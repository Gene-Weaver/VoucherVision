import os
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(layout="wide", page_icon='img/icon.ico', page_title='VV FAQs',initial_sidebar_state="collapsed")

def display_faqs():
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
        st.write('If you would like to get more involved, have questions, would like to see additional features, then please fill out this [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSe2E9zU1bPJ1BW4PMakEQFsRmLbQ0WTBI2UXHIMEFm4WbnAVw/viewform?usp=sf_link)')
        components.iframe(f"https://docs.google.com/forms/d/e/1FAIpQLSe2E9zU1bPJ1BW4PMakEQFsRmLbQ0WTBI2UXHIMEFm4WbnAVw/viewform?embedded=true", height=900,scrolling=True,width=640)

    with c1:
        st.header('FAQs')
        st.subheader('Lead Institution')
        st.write('- University of Michigan')

        st.subheader('Partner Institutions')
        st.write('- Oregon State University')
        st.write('- University of Colorado Boulder')
        st.write('- Botanical Research Institute of Texas')
        st.write('- South African National Biodiversity Institute')
        st.write('- Smithsonian National Museum of Natural History')
        st.write('- National Botanic Gardens of Ireland')
        st.write('- Center for Wood Anatomy Research')
        st.write('- Botanischer Garten Berlin')
        st.write('- Freie Universität Berlin')
        st.write('- Cambridge University')
        st.write('- Morton Arboretum')
        st.write('- Florida Museum')
        st.write('- iDigBio')
        st.write('**More soon!**')

display_faqs()