# %% Imports
import json
import pandas as pd
import random
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from dotenv import load_dotenv
from yaml.loader import SafeLoader
from helpers.helper import FinQA

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


# %% Main App Body
if st.session_state["authentication_status"]:

    # %% Initialise
    # Load (Pre-processed) Data
    data = load_data()
    # Load the .env file
    load_dotenv()

    # %% Tabs
    (
        tab_sample_review,
        tab_accuracy,
    ) = st.tabs(
        [
            "Sample Review",
            "Accuracy Report",
        ]
    )

    with tab_sample_review:
        st.markdown(
            "Welcome to the [ConvFinQA](https://github.com/czyssrs/ConvFinQA) data "
            "preview. The data has been slightly preprocessed for better usability, "
            "and in this tab, you can explore samples of the data.\n\n"
            "Use the **Randomize Data** button to view a random sample, including "
            "its content along with the corresponding question and answer.\n\n"
        )
        # select a random index
        total_data = 3965
        if "random_idx" not in st.session_state:
            st.session_state["random_idx"] = random.randint(0, total_data - 1)
        if st.button("Randomize Data"):
            st.session_state["random_idx"] = random.randint(0, total_data - 1)
        random_idx = st.session_state["random_idx"]

        # Show Sample Data
        st.markdown("---\n\n## Sample Data")
        st.markdown(
            f"There are **{total_data}** data points. Showing data for index: **{random_idx}**"
        )
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

        # Run the model and Show LLM Output
        st.markdown(
            "---\n\n## Run Language Model\n\n"
            "To see the output of the model for the above question, click "
            "**Run LLM** to generate a result from the language model."
        )
        # Model Selection
        gpt_model = st.selectbox(
            options=["gpt-4o-mini", "gpt-3.5-turbo"], label="GPT Model"
        )
        if st.button("Run LLM"):
            with st.spinner("Running LLM..."):
                try:
                    fin_qa = FinQA(gpt_model)
                    answer, responder = fin_qa.get_answer(
                        pre_text=" ".join(data[random_idx]["pre_text"]),
                        table=data[random_idx]["table"],
                        post_text=" ".join(data[random_idx]["post_text"]),
                        question=data[random_idx]["question"],
                    )
                    st.markdown(f"**{responder}**:\n\n {answer}")
                except Exception as e:
                    st.error(f"LLM not able to geenrate output. Error: {e}")

    with tab_accuracy:
        pass
