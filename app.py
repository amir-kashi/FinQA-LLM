import json
import pandas as pd
import random
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

# Site Header
st.markdown(
    """
# Financial Question Answering with Large Language Models 

This is a demo of a financial question answering system using large language models.
"""
)

# %% Authentication
with open(".streamlit/auth.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

name, authentication_status, username = authenticator.login("main")

if st.session_state["authentication_status"]:
    st.sidebar.write(f'Welcome *{st.session_state["name"]}*')
    authenticator.logout("Logout", "sidebar")
elif st.session_state["authentication_status"] == False:
    st.error("Username/password is incorrect")
elif st.session_state["authentication_status"] == None:
    st.warning("Please enter your username and password")


# %% Functions


@st.cache_data(ttl=10, show_spinner=True)
def load_data():
    with open("data/extracted_data.json", "r") as f:
        data = json.load(f)
    return data


# %% Load (Pre-processed) Data
data = load_data()

# %% Main App Body
if st.session_state["authentication_status"]:

    # %% Sidebar
    st.sidebar.header("User Settings")
    st.sidebar.markdown("""---""")
    # 1. Show random data
    total_data = len(data)
    random_idx = random.randint(0, total_data - 1)
    st.sidebar.markdown(
        f"here are **{total_data}** data points. Showing data for index **{random_idx}**"
    )
    st.sidebar.button("Randomize Data")
    # 2. Model Selection
    gpt_model = st.sidebar.selectbox(
        options=["gpt-4o-mini", "gpt-3.5-turbo"], label="GPT Model"
    )

    # %% Tabs
    (
        tab_data_prevview,
        tab_llm_output,
    ) = st.tabs(
        [
            "Data Preview",
            "LLM Output",
        ]
    )

    with tab_data_prevview:
        # dict_keys(['id', 'pre_text', 'post_text', 'filename', 'table_ori', 'table', 'annotation', 'question', 'answer'])
        st.markdown(f"**ID**: {data[random_idx]['id']}")
        st.markdown(f"**Filename**: {data[random_idx]['filename']}")
        with st.expander("**Pre Text**"):
            st.markdown(" ".join(data[random_idx]["pre_text"]))
        with st.expander("**Post Text**"):
            st.markdown(" ".join(data[random_idx]["post_text"]))
        with st.expander("**Original Table**"):
            table = data[random_idx]["table_ori"]
            try:
                st.dataframe(pd.DataFrame(columns=table[0], data=table[1:]))
            except:
                st.write(table)
        with st.expander("**Table**"):
            table = data[random_idx]["table"]
            try:
                st.dataframe(pd.DataFrame(columns=table[0], data=table[1:]))
            except:
                st.write(table)
        with st.expander("**Annotation**"):
            st.write(data[random_idx]["annotation"])

        st.markdown(f"**Question**:\n\n{data[random_idx]['question']}")
        st.markdown(f"**Answer**:\n\n {data[random_idx]['answer']}")

    with tab_llm_output:
        pass
