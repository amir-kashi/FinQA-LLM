# %% Imports
import json
import pandas as pd
import plotly.express as px
import random
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from dotenv import load_dotenv
from yaml.loader import SafeLoader
from helpers.helper import FinQA

# %% App Config
st.set_page_config(
    page_title="Conv Fin QA - tomoro.ai",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto",
)

# %% Site Header
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


@st.cache_data(ttl=600, show_spinner=True)
def load_data():
    with open("data/extracted_data.json", "r") as f:
        data = json.load(f)
    return data


@st.cache_data(ttl=600, show_spinner=True)
def load_accuracy_df():
    df_with_score = pd.read_csv("data/df_with_score.csv", index_col=False)
    return df_with_score


# %% Main App Body
if st.session_state["authentication_status"]:

    # %% Initialise
    # Load (Pre-processed) Data
    data = load_data()
    # Load Accuracy Data
    df_with_score = load_accuracy_df()
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
        st.markdown(
            "Explore the accuracy of the model by applying filters to the data. "
            "The number of remaining samples and the bar chart will update "
            "dynamically based on your selections."
        )

        # Create 2 columns
        col1, col2 = st.columns([1, 1], gap="small")

        with col1:  # Settings
            st.markdown(
                "### Filter Data\n\nApply filters to refine the dataset and see "
                "how the accuracy changes. The bar chart on the right will "
                "update to reflect your filtered data."
            )

            filter_no_answer = st.checkbox("Filter samples with **No Answer**")
            filter_time_out = st.checkbox("Filter cases where the LLM was Time Out")
            filter_no_context = st.checkbox("Filter cases where No Context was found")

            # Filter Data
            filtered_df = df_with_score.copy()
            if filter_no_answer:
                filtered_df = filtered_df.dropna(subset=["answer"])
            if filter_time_out:
                filtered_df = filtered_df[
                    filtered_df["llm_answer"]
                    != "Agent stopped due to iteration limit or time limit."
                ]
            if filter_no_context:
                filtered_df = filtered_df[
                    filtered_df["llm_answer"] != "Not Enough Context"
                ]
            st.markdown(
                f"**Filtered Data**: {len(filtered_df)} out of {len(df_with_score)}"
            )

            # Group by Responder and Score
            df_grouped = (
                filtered_df.groupby(["responder", "score"])
                .size()
                .reset_index(name="count")
            )

            # Accuracy results
            total_agent = df_grouped[df_grouped["responder"] == "AGENT RESPONSE"][
                "count"
            ].sum()
            total_llm = df_grouped[df_grouped["responder"] == "LLM RESPONSE"][
                "count"
            ].sum()
            total = total_agent + total_llm
            accuracy_agent = (
                df_grouped[
                    (df_grouped["responder"] == "AGENT RESPONSE")
                    & (df_grouped["score"] == True)
                ]["count"].sum()
                / total_agent
            ) * 100

            accuracy_llm = (
                df_grouped[
                    (df_grouped["responder"] == "LLM RESPONSE")
                    & (df_grouped["score"] == True)
                ]["count"].sum()
                / total_llm
            ) * 100

            accuracy_total = (
                df_grouped[df_grouped["score"] == True]["count"].sum()
                / df_grouped["count"].sum()
            ) * 100

            st.markdown(
                "#### Accuracy Results\n\n"
                f"| Response Type     | No. of Cases  | Accuracy                 |\n"
                f"|-------------------|---------------|--------------------------|\n"
                f"| Agent Response    | {total_agent} | {accuracy_agent:.2f}%    |\n"
                f"| LLM Response      | {total_llm}   | {accuracy_llm:.2f}%      |\n"
                f"| **Total**         | **{total}**   | **{accuracy_total:.2f}%**|\n"
            )

        with col2:  # Data
            # Create the grouped bar chart
            fig = px.bar(
                df_grouped,
                x="responder",
                y="count",
                color="score",
                title="Comparison of AGENT RESPONSE vs LLM RESPONSE based on Score",
                labels={
                    "count": "Count of Responses",
                    "responder": "Responder",
                    "score": "Score",
                },
                barmode="group",
                color_discrete_map={True: "lightgreen", False: "orange"},
                text_auto=True,
            )

            fig.update_layout(
                xaxis_title="Responder",
                yaxis_title="Count",
                legend_title="Score",
                bargap=0.15,  # Gap between bars of adjacent location coordinates.
            )
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)

        # Accuracy report
        with open("data/accuracy_report.md", "r") as f:
            accuracy_report = f.read()
            with st.expander("Show Accuracy Report"):
                st.markdown(accuracy_report)
