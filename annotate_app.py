import streamlit as st
import pandas as pd
import os
from os.path import join, exists

# Example: 'data/experiments/llmq-gpt-4o-mini/llmr-gpt-3.5/docp-dt03'
experiment_folder = "data/experiments/llmq-gpt-4o-mini/llmr-gpt-3.5/docp-dt-z-1"

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

def init():
    # Set the layout to wide to make use of the full screen width
    st.set_page_config(layout="wide")
    cwd = os.getcwd() 
    # Title of the app
    st.title("Defuse Project Data Labeling")
    # Initialize session state for document and question content
    if "document_content" not in st.session_state:
        st.session_state.document_content = ""
    if "question_content" not in st.session_state:
        st.session_state.question_content = ""
        
    # Inject custom CSS to make the left column sticky
    st.markdown(
        """
        <style>
        /* Make the left column sticky */
        div[data-testid="column"]:nth-child(1) {
            position: sticky;
            top: 0;
            height: 100vh;
            overflow-y: auto;
            padding-right: 10px;
            padding-bottom: 30px;
            border-right: 2px solid #ccc;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    return cwd 

def sidebar_logic(cwd, experiment_folder):
    # Sidebar for selecting Experiment and Topic
    st.sidebar.header("Which Experiment/Topic to Work On")
    base_path = join(cwd, experiment_folder)
    # Get the list of experiment folders 
    try:
        exp_folders = [f for f in os.listdir(base_path) if os.path.isdir(join(base_path, f))]
    except FileNotFoundError:
        st.error("The base path does not exist.")
        exp_folders = []
    # Dropdown for selecting experiment folder
    if exp_folders:
        exp_name = st.sidebar.selectbox("Choose Experiment Name:", exp_folders)
        experiment_dir = join(base_path, exp_name)
    else:
        st.error("No experiments found in the base path.")
        st.stop()

    # Get the list of topic folders
    try: 
        topic_folders = [f for f in os.listdir(experiment_dir) if os.path.isdir(join(experiment_dir, f))]
    except FileNotFoundError:
        st.error("The experiment path does not exist.")
        topic_folders = []
    # Dropdown for selecting topic folder
    if topic_folders: 
        topic = st.sidebar.selectbox("Choose Topic:", topic_folders)
        data_dir = join(experiment_dir, topic)
    else:
        st.error("No topics found in the experiment path")
        st.stop()
        
    return exp_name, data_dir

def check_username_csv_path(cwd, exp_name):
    # Check if annotator_name is in session_state
    if 'annotator_name' not in st.session_state:
        annotator_name = st.text_input("Enter your Username:", key="annotator_name_input")
        if annotator_name:
            st.session_state.annotator_name = annotator_name
    else:
        annotator_name = st.session_state.annotator_name

    if not annotator_name:
        st.warning("Please enter your Username to proceed.")
        st.stop()
    
    csv_filename = f"{annotator_name}_{exp_name}_labels.csv"
    csv_path = join(cwd, csv_filename)
    check_and_create_annotations_csv(csv_path)
    
    return csv_path

def check_and_create_annotations_csv(csv_path):
    # Check if the CSV file exists
    if not exists(csv_path):
        st.warning(f"CSV does not yet exist: {csv_path}")

        # Check if the user has made a choice already
        if 'create_csv_choice' not in st.session_state:
            st.write("### Do you want to create this new annotation CSV file? Make sure you have entered the correct Experiment Name and Username")
            choice = st.selectbox(
                "Please select an option:",
                ("Select an option", "Yes, create the file", "No, do not create")
            )
            if choice == "Yes, create the file":
                st.session_state.create_csv_choice = 'Yes'
                st.rerun()
            elif choice == "No, do not create":
                st.session_state.create_csv_choice = 'No'
                st.rerun()
            else:
                st.stop()
        else:
            if st.session_state.create_csv_choice == 'Yes':
                # Create the DataFrame with the specified columns
                columns = [
                    'doc_id', 'q_id', 'supposed_to_be_confusing', 'llm_confuse_label',
                    'llm_defuse_label', 'human_confuse_label', 'human_defuse_label',
                    'question_category'
                ]
                annotations_df = pd.DataFrame(columns=columns)
                # Save the DataFrame as a CSV file
                annotations_df.to_csv(csv_path, index=False)
                st.success(f"{csv_path} created successfully.")
                return csv_path
            else:
                st.warning("No CSV file created. Cannot proceed without a CSV file. Refresh to restart")
                st.stop()
    else:
        st.success(f"Annotations will be saved to: \"{csv_path}\"")

def select_doc_id_with_checkmarks(doc_data, qrc_data, annotations_df):
    doc_ids = doc_data["doc_id"].unique()
    
    doc_id_labels = []
    doc_id_mapping = {}
    for doc_id in doc_ids:
        is_fully_annotated = check_if_document_fully_annotated(annotations_df, doc_id, qrc_data, return_bool=True)
        if is_fully_annotated:
            label = f"✅ {doc_id}"
        else:
            label = f"{doc_id}"
        doc_id_labels.append(label)
        doc_id_mapping[label] = doc_id

    selected_label = st.sidebar.selectbox("Choose doc_id:", doc_id_labels)
    doc_id = doc_id_mapping[selected_label]
    return doc_id

def show_instructions():
    st.write("### Instructions:")
    st.write('''Make sure Experiment, Topic, and doc_id is correct. Read the "Document", take your time and understand what it's talking about''')
    st.write('''"Question" is a confusing question generated by LLM_q (gpt4o-mini) after a series of hallucination steps. ''')
    st.write('''"Response" is LLM_r (gpt3.5)'s response after prompted with <Document, Question, "Read the document and answer the question">''')
    st.write('''Fill out the form for each question, then click "Submit", after finishing all questions for this document, move on to the next document by selecting "Choose doc_id" on the left sidebar''')

def show_doc_contents(doc_data, doc_id):
    st.session_state.document_content = doc_data[doc_data["doc_id"] == doc_id]["document"].values[0]
    # Display Document Content
    if st.session_state.document_content:
        st.write(f"### Document: {doc_id}")
        st.write(st.session_state.document_content)

# Function to append a row to the CSV file
def append_row_to_csv(csv_path, row_data, qrc_data):
    annotations_df = pd.read_csv(csv_path)
    existing_entry_index = annotations_df[
        (annotations_df['doc_id'] == row_data['doc_id']) &
        (annotations_df['q_id'] == row_data['q_id']) & 
        (annotations_df['supposed_to_be_confusing'] == row_data['supposed_to_be_confusing'])
    ].index
    
    if not existing_entry_index.empty:
        # Overwrite the existing entry
        # Create a DataFrame from row_data with the same index
        new_row_df = pd.DataFrame([row_data], columns=annotations_df.columns, index=existing_entry_index)
        annotations_df.loc[existing_entry_index] = new_row_df
        annotations_df.to_csv(csv_path, index=False)
        st.info(f"Overwritten previous annotation.")
    else:
        # Append the new annotation
        new_row = pd.DataFrame([row_data], columns=annotations_df.columns)
        annotations_df = pd.concat([annotations_df, new_row], ignore_index=True)
        annotations_df.to_csv(csv_path, index=False)
        st.success(f"Annotation submitted.")
        
    check_if_document_fully_annotated(annotations_df, row_data['doc_id'], qrc_data)

def check_if_document_fully_annotated(annotations_df, doc_id, qrc_data, return_bool=False):
    selected_qrc = qrc_data[(qrc_data["doc_id"] == doc_id)]
    selected_qrc = selected_qrc.sample(frac=1, random_state=42).reset_index(drop=True)
    
    question_mapping = {}
    for index, row in selected_qrc.iterrows():
        question_mapping[(row['q_id'], row['is_confusing'])] = index + 1

    # Create a set of unique identifiers for annotated questions
    annotated_questions = set(
        zip(
            annotations_df[annotations_df['doc_id'] == doc_id]['q_id'],
            annotations_df[annotations_df['doc_id'] == doc_id]['supposed_to_be_confusing']
        )
    )
    # Create a set of unique identifiers for all questions in the selected_qrc
    all_questions = set(
        zip(
            selected_qrc['q_id'],
            selected_qrc['is_confusing']
        )
    )
    remaining_questions = all_questions - annotated_questions
    if not remaining_questions:
        if return_bool:
            return True
        st.success(f"All questions for Document ID {doc_id} have been annotated.")
    else:
        if return_bool:
            return False
        remaining_question_indexes = [question_mapping[question] for question in remaining_questions]
        st.info(f"Question # not yet annotated: {sorted(remaining_question_indexes)} for Document {doc_id}")

def show_question_contents_and_annotation_form(qrc_data, doc_id, csv_path, annotations_df):
    # Select all questions associated with this document
    selected_qrc = qrc_data[(qrc_data["doc_id"] == doc_id)]
    # Shuffle
    selected_qrc = selected_qrc.sample(frac=1, random_state=42).reset_index(drop=True)

    if not selected_qrc.empty:
        for index, row in selected_qrc.iterrows():
            q_id = row['q_id']
            supposed_to_be_confusing = row['is_confusing']
            st.write(f"**Question #{index + 1}**:")  # This is NOT the actual q_id, it's just here so annotators know where they are at.
            llm_confuse_label = row['confusion'].split("\n")[0]
            llm_defuse_label = row['is_defused']
            st.text_area("Question:", value=row['question'], key=f"question_{index}")
            st.text_area("Response:", value=row['response'], key=f"response_{index}")

            with st.form(key=f'annotation_form_{index}', clear_on_submit=True):
                st.write("##### Your Annotations:")
                human_confuse_label_options = ["Did not select", "Yes", "No"]
                human_confuse_label = st.radio(
                    "Is this question confusing? (Please select Yes or No)",
                    human_confuse_label_options,
                    key=f"human_confuse_label_{index}",
                )
                question_category_options = ['False Premise/Assumption', 'Not Mentioned/Relevant', 'Ambiguous', 'Other']
                question_category = st.multiselect(
                    "Select the category of confusion (usually 1 category is enough):",
                    options=question_category_options,
                    key=f"question_category_{index}"
                )
                # If 'Other' is selected, display a text input for the custom category
                other_category = ""
                if 'Other' in question_category:
                    other_category = st.text_input(
                        "Please specify the other category:",
                        key=f"other_category_{index}"
                    )
                human_defuse_label_options = ["Did not select", "Yes", "No"]
                human_defuse_label = st.radio(
                    "Did the LLM's response defuse the confusion? Select \"Did not select\" if the question is not confusing in the first place",
                    human_defuse_label_options,
                    key=f"human_defuse_label_{index}",
                )
                submit_button = st.form_submit_button(label='Submit')
                if submit_button:
                    # Handle 'Other' category
                    if 'Other' in question_category:
                        if other_category:
                            question_category.remove('Other')  # Remove 'Other' placeholder
                            question_category.append(other_category)  # Add the custom category
                        else:
                            st.error("Please specify the 'Other' category.")
                            st.stop()
                            
                    question_category_str = ', '.join(question_category) if question_category else "Did not select"

                    # When the submit button is clicked, append the data to the CSV
                    row_data = {
                        'doc_id': doc_id,
                        'q_id': q_id,
                        'supposed_to_be_confusing': supposed_to_be_confusing,
                        'llm_confuse_label': llm_confuse_label,
                        'llm_defuse_label': llm_defuse_label,
                        'human_confuse_label': human_confuse_label,
                        'human_defuse_label': human_defuse_label,
                        'question_category': question_category_str
                    }
                    append_row_to_csv(csv_path, row_data, qrc_data)
                    
                    
            st.write("---")  # Add a separator between questions
    else:
        st.write("No data found for the selected document and confusion status.")

######## Script Below ###########

cwd = init() 

# Sidebar logic to select experiment and topic
exp_name, data_dir = sidebar_logic(cwd, experiment_folder)

# Now we can get the CSV path since we have exp_name
csv_path = check_username_csv_path(cwd, exp_name)

# Load data
doc_data, qrc_data, qrc_filter_data = load_csv_data(data_dir)

# Load annotations DataFrame
if exists(csv_path):
    annotations_df = pd.read_csv(csv_path)
else:
    columns = [
        'doc_id', 'q_id', 'supposed_to_be_confusing', 'llm_confuse_label',
        'llm_defuse_label', 'human_confuse_label', 'human_defuse_label',
        'question_category'
    ]
    annotations_df = pd.DataFrame(columns=columns)

# Select doc_id with checkmarks
doc_id = select_doc_id_with_checkmarks(doc_data, qrc_data, annotations_df)

left, right = st.columns([2 , 1.5])  # these numbers represent proportions

with left:
    show_instructions()
    show_doc_contents(doc_data, doc_id)
    
with right:
    show_question_contents_and_annotation_form(qrc_data, doc_id, csv_path, annotations_df)