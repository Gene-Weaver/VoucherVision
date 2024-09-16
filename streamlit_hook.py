import os
import sys

def runtime_hook():
    if getattr(sys, 'frozen', False):
        os.environ['STREAMLIT_STATIC_PATH'] = os.path.join(sys._MEIPASS, 'streamlit', 'static')

runtime_hook()