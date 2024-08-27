import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader


# %% Authentication
with open(".streamlit/auth.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

# Header
st.markdown(
    """
# Financial Question Answering with Large Language Models 

This is a demo of a financial question answering system using large language models.
"""
)

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
    # config["preauthorized"],
)

# authenticator.login("Login", "main")
name, authentication_status, username = authenticator.login("main")
print(name, authentication_status, username)


if st.session_state["authentication_status"]:
    st.sidebar.write(f'Welcome *{st.session_state["name"]}*')
    authenticator.logout("Logout", "sidebar")
elif st.session_state["authentication_status"] == False:
    st.error("Username/password is incorrect")
elif st.session_state["authentication_status"] == None:
    st.sidebar.warning("Please enter your username and password")
