import streamlit.web.cli as stcli
import os, sys

# Insert a file uploader that accepts multiple files at a time:
# import streamlit as st
# uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
# for uploaded_file in uploaded_files:
#     bytes_data = uploaded_file.read()
#     st.write("filename:", uploaded_file.name)
#     st.write(bytes_data)


def resolve_path(path):
    resolved_path = os.path.abspath(os.path.join(os.getcwd(), path))
    return resolved_path


if __name__ == "__main__":
    dir_home = os.path.dirname(__file__)

    # pip install protobuf==3.20.0

    sys.argv = [
        "streamlit",
        "run",
        resolve_path(os.path.join(dir_home,"vouchervision", "VoucherVision_GUI.py")),
        "--global.developmentMode=false",
        "--server.port=8526",

    ]
    sys.exit(stcli.main())