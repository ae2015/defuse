import streamlit as st
import pandas as pd
import os
from os.path import join

# Set the layout to wide to make use of the full screen width
st.set_page_config(layout="wide")

cwd = os.getcwd()

topics = [
    'sport', 'business', 'science', 'finance', 'food',
    'politics', 'economics', 'travel', 'entertainment',
    'music', 'news'
]

# Load CSV files
@st.cache_data
def load_csv_data(data_dir):
    doc_data = pd.read_csv(join(data_dir, "docs_out.csv"))  # Load document CSV
    qrc_data = pd.read_csv(join(data_dir, "qrc_out.csv"))  # Load QRC CSV
    try:
        qrc_filter_data = pd.read_csv(join(data_dir, "qrc_filter.csv"))
        if qrc_filter_data.empty:
            st.warning("The qrc_filter.csv file is empty.")
    except pd.errors.EmptyDataError:
        qrc_filter_data = pd.DataFrame()
        st.warning("The qrc_filter.csv file is blank or could not be read.")
    return doc_data, qrc_data, qrc_filter_data

# Initialize session state for document and question content
if "document_content" not in st.session_state:
    st.session_state.document_content = ""

if "question_content" not in st.session_state:
    st.session_state.question_content = ""

# Define the document attributes available for selection
doc_attrs = ["source", "document", "LLM_q", "doc_prompt", "reduce_doc", "modify_doc", "orig_questions", "conf_questions"]

# Title of the app
st.title("Data Viewer")

# Sidebar for doc_id selection
st.sidebar.header("Document")
default_exp_name = "20-toy-dev"
exp_name = st.text_input("Experiment Name: ", value=default_exp_name)
experiment_dir = join(cwd, 'data/experiments/llmq-gpt-4o-mini/llmr-gpt-3.5/docp-dt-z-1', exp_name)
topic = st.sidebar.selectbox("Choose topic:", topics)

# Read the CSV files
data_dir = join(experiment_dir, topic)
doc_data, qrc_data, qrc_filter_data = load_csv_data(data_dir)
doc_id = st.sidebar.selectbox("Choose doc_id:", doc_data["doc_id"].unique())

# Sidebar for doc_attr selection
doc_attr = st.sidebar.selectbox("Choose document attribute to display:", doc_attrs)

# Sidebar for question confusion status selection
st.sidebar.header("Question")
is_confusing = st.sidebar.selectbox("Choose is_confusing:", ["no", "yes"])

# Show Document Content
if st.button("Show Document Content"):
    st.session_state.document_content = doc_data[doc_data["doc_id"] == doc_id][doc_attr].values[0]

# Show Question Content (Revised to match the format)
if st.button("Show Question Content"):
    selected_qrc = qrc_data[(qrc_data["doc_id"] == doc_id) & (qrc_data["is_confusing"] == is_confusing)]

    if not selected_qrc.empty:
        for index, row in selected_qrc.iterrows():
            st.write(f"**Question ID**: {row['q_id']} | Confusion: {row['confusion']}, is_defused: {row['is_defused']}")
            st.text_area("Question:", value=row['question'], height=100, key=f"question_{index}")
            st.text_area("Response:", value=row['response'], height=100, key=f"response_{index}")
            st.text_area("Defusion:", value=row['defusion'], height=100, key=f"defusion_{index}")
            st.write("---")  # Add a separator between entries
    else:
        st.write("No data found for the selected document and confusion status.")

# Display Document Content
if st.session_state.document_content:
    st.subheader(f"Document ID: {doc_id}")
    st.text_area(f"Content of {doc_attr}:", value=st.session_state.document_content, height=300)

# Display Question Content (Multiple questions and responses are displayed individually)
if st.session_state.question_content:
    st.subheader("Questions and Responses")
    st.text_area("Question Content:", value=st.session_state.question_content, height=300)