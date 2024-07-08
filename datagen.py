import pandas as pd
import utils, promptlib
from tqdm import tqdm

def generate_questions_for_documents(llm, num_q, schema, path_in, path_out):
    df_in = utils.read_csv(path_in, "Read the input document table from CSV file")

    # Check that the correct columns are present in the CSV
    assert {
            schema["doc_id"],
            schema["source"],
            schema["document"]
        }.issubset(set(df_in.columns))
    assert {schema["LLM_q"], schema["orig_qs"]}.isdisjoint(set(df_in.columns))

    print(f"Generate {num_q} questions for each document")
    rows_out = []
    for _, row in tqdm(df_in.iterrows(), total = df_in.shape[0]):
        doc_id = row[schema["doc_id"]]
        doc_source = row[schema["source"]]
        raw_document = row[schema["document"]]
        document = utils.prepare_document(raw_document)
        questions = promptlib.generate_questions(llm, document, num_q)
        row_out = {
            schema["doc_id"] : doc_id,
            schema["source"] : doc_source,
            schema["document"] : document,
            schema["LLM_q"] : llm,
            schema["orig_qs"] : "\n".join([f"{i}. {q}" for i, q in enumerate(questions, start = 1)])
        }
        rows_out.append(row_out)

    df_out = pd.DataFrame.from_dict(rows_out, dtype = str)
    utils.write_csv(df_out, path_out, "Write the table with questions to CSV file")

def infuse_questions_with_false_assumptions(schema, path_in, path_out):
    df_in = utils.read_csv(path_in, "Read the document-and-questions table from CSV file")
    print("Modify each question by adding confusing (false) assumptions")
    rows_out = []
    for _, row in tqdm(df_in.iterrows(), total = df_in.shape[0]):
        doc_id = row[schema["doc_id"]]
        doc_source = row[schema["source"]]
        document = row[schema["document"]]
        llm = row[schema["LLM_q"]]
        orig_questions = utils.parse_numbered_list(row[schema["orig_qs"]])
        conf_questions = promptlib.confuse_questions(llm, document, orig_questions)
        row_out = {
            schema["doc_id"] : doc_id,
            schema["source"] : doc_source,
            schema["document"] : document,
            schema["LLM_q"] : llm,
            schema["orig_qs"] : "\n".join([f"{i}. {q}" for i, q in enumerate(orig_questions, start = 1)]),
            schema["conf_qs"] : "\n".join([f"{i}. {q}" for i, q in enumerate(conf_questions, start = 1)])
        }
        rows_out.append(row_out)

    df_out = pd.DataFrame.from_dict(rows_out, dtype = str)
    utils.write_csv(df_out, path_out, "Write the table with confusing questions to CSV file")

def generate_RAG_responses(llm, doc_schema, doc_path, qr_schema, qr_path):
    df_in = utils.read_csv(doc_path, "Read the document-and-questions table from CSV file")
    print("Generate RAG response for each question, both original and confusing")
    rows_out = []
    for _, row in tqdm(df_in.iterrows(), total = df_in.shape[0]):
        doc_id = row[doc_schema["doc_id"]]
        document = row[doc_schema["document"]]
        orig_questions = utils.parse_numbered_list(row[doc_schema["orig_qs"]])
        conf_questions = utils.parse_numbered_list(row[doc_schema["conf_qs"]])
        assert len(orig_questions) == len(conf_questions)
        for q_id, (orig_q, conf_q) in enumerate(zip(orig_questions, conf_questions), start = 1):
            response_orig_q = promptlib.generate_response(llm, document, orig_q)
            row_out = {
                qr_schema["doc_id"] : doc_id,
                qr_schema["q_id"] : q_id,
                qr_schema["is_conf"] : "no",
                qr_schema["question"] : orig_q,
                qr_schema["LLM_r"] : llm,
                qr_schema["response"] : response_orig_q
            }
            rows_out.append(row_out)
            response_conf_q = promptlib.generate_response(llm, document, conf_q)
            row_out = {
                qr_schema["doc_id"] : doc_id,
                qr_schema["q_id"] : q_id,
                qr_schema["is_conf"] : "yes",
                qr_schema["question"] : conf_q,
                qr_schema["LLM_r"] : llm,
                qr_schema["response"] : response_conf_q
            }
            rows_out.append(row_out)

    df_out = pd.DataFrame.from_dict(rows_out, dtype = str)
    utils.write_csv(df_out, qr_path, "Write the question-response table to CSV file")

def create_dictionary_of_indexed_documents(doc_schema, doc_path):
    df_doc = utils.read_csv(doc_path, "Read the document table from CSV file")
    print("Create a dictionary of indexed documents")
    documents = {
        row[doc_schema["doc_id"]] : row[doc_schema["document"]]
            for _, row in tqdm(df_doc.iterrows(), total = df_doc.shape[0])
    }
    return documents

def find_false_assumptions_in_questions(doc_schema, doc_path, qr_schema, qr_path_in, qr_path_out):
    documents = create_dictionary_of_indexed_documents(doc_schema, doc_path)
    df_qr = utils.read_csv(qr_path_in, "Read the question-response table from CSV file")    
    print("Ask LLM to find a false assumption in each question, or say 'none'")
    rows_out = []
    for _, row in tqdm(df_qr.iterrows(), total = df_qr.shape[0]):
        doc_id = row[qr_schema["doc_id"]]
        document = documents[doc_id]
        question = row[qr_schema["question"]]
        llm = row[qr_schema["LLM_r"]]
        confusion = promptlib.find_false_assumption(llm, document, question)
        row_out = dict(row)
        row_out[qr_schema["confusion"]] = confusion
        rows_out.append(row_out)
    df_out = pd.DataFrame.from_dict(rows_out, dtype = str)
    utils.write_csv(df_out, qr_path_out, "Write the question-response table to CSV file")

def check_if_response_defused_confusion(doc_schema, doc_path, qr_schema, qr_path_in, qr_path_out):
    documents = create_dictionary_of_indexed_documents(doc_schema, doc_path)
    df_qr = utils.read_csv(qr_path_in, "Read the question-response table from CSV file")
    print("Ask LLM to check if its own response defused the confusion")
    rows_out = []
    for _, row in tqdm(df_qr.iterrows(), total = df_qr.shape[0]):
        doc_id = row[qr_schema["doc_id"]]
        document = documents[doc_id]
        question = row[qr_schema["question"]]
        llm = row[qr_schema["LLM_r"]]
        response = row[qr_schema["response"]]
        confusion = row[qr_schema["confusion"]]
        if confusion == "none":
            defusion, is_defused = "n/a", "n/a"
        else:
            defusion, is_defused = promptlib.check_response_for_defusion(llm, document, question, response, confusion)
        row_out = dict(row)
        row_out[qr_schema["defusion"]] = defusion
        row_out[qr_schema["is_defused"]] = is_defused
        rows_out.append(row_out)
    df_out = pd.DataFrame.from_dict(rows_out, dtype = str)
    utils.write_csv(df_out, qr_path_out, "Write the question-response table to CSV file")

def filter_undefused_confusions_and_compute_metrics(qr_schema, qr_path, filter_path):
    df_qr = utils.read_csv(qr_path, "Read the question-response table from CSV file")

    num_orig_questions = 0
    num_conf_questions = 0
    num_orig_questions_with_confusion_detected = 0
    num_conf_questions_with_confusion_detected = 0
    num_orig_questions_with_conf_detected_and_defused = 0
    num_conf_questions_with_conf_detected_and_defused = 0

    filter_rows = []
    for _, row in tqdm(df_qr.iterrows(), total = df_qr.shape[0]):
        is_conf = row[qr_schema["is_conf"]]
        confusion = row[qr_schema["confusion"]]
        is_defused = row[qr_schema["is_defused"]]
        if is_conf == "yes":
            num_conf_questions += 1
            if confusion != "none":
                num_conf_questions_with_confusion_detected += 1
                if is_defused == "yes":
                    num_conf_questions_with_conf_detected_and_defused += 1
                else:
                    filter_rows.append(dict(row))
        elif is_conf == "no":
            num_orig_questions += 1
            if confusion != "none":
                num_orig_questions_with_confusion_detected += 1
                if is_defused == "yes":
                    num_orig_questions_with_conf_detected_and_defused += 1
    print("Original (non-confusing) questions:")
    print(f"    Total questions = {num_orig_questions}")
    print(f"    With confusion detected = {num_orig_questions_with_confusion_detected}")
    print(f"    With confusion detected and defused = {num_orig_questions_with_conf_detected_and_defused}")
    print("Confusing questions:")
    num_conf_questions_with_conf_detected_but_undefused = \
        num_conf_questions_with_confusion_detected - num_conf_questions_with_conf_detected_and_defused
    print(f"    Total questions = {num_conf_questions}")
    print(f"    With confusion detected = {num_conf_questions_with_confusion_detected}")
    print(f"    With confusion detected and defused = {num_conf_questions_with_conf_detected_and_defused}")
    print(f"    With confusion detected, but not defused = {num_conf_questions_with_conf_detected_but_undefused}")

    df_filter = pd.DataFrame.from_dict(filter_rows, dtype = str)
    utils.write_csv(df_filter, filter_path, "Write the filtered question-response table to CSV file")



if __name__ == "__main__":

    doc_csv_schema = {
        "doc_id" : "doc_id",           # Column with a unique document ID
        "source" : "source",           # Column with document source (e.g. URL)
        "document" : "document",       # Column with the text of the original document
        "LLM_q" : "LLM_q",             # Column with name of the LLM that generated confusing questions
        "orig_qs" : "orig_questions",  # Column with the original (non-confusing) questions
        "conf_qs" : "conf_questions"   # Column with the modified (confusing) questions
    }

    qrc_csv_schema = {
        "doc_id" : "doc_id",           # Column with document ID, same as the other table
        "q_id" : "q_id",               # Column with question ID (for this document)
        "is_conf" : "is_confusing",    # Column with "yes" or "no" indicating if the question is confusing
        "question" : "question",       # Column with the question (either original or confusing)
        "LLM_r" : "LLM_r",             # Column with name of the LLM that generated the responses
        "response" : "response",       # Column with response generated given the document and the question
        "confusion" : "confusion",     # Column with LLM-found confusion in the question (or "none")
        "defusion" : "defusion",       # Column with LLM's reply on whether its own response detected the confusion
        "is_defused" : "is_defused"    # Column with "yes" or "no" as LLM checks if response detected the confusion
    }

    # Note: LLM_q may be stronger than LLM_r, to create more challenging confusions.

    llm_q = "gpt-3.5"  # LLM for generating questions (stronger)
    llm_r = "gpt-3.5"  # LLM for generating responses (weaker)

    num_q = 10  # Number of questions per document

    doc_path_0 = "docs.csv"
    doc_path_1 = "docs_1.csv"
    doc_path_2 = "docs_2.csv"
    qrc_path_1 = "qrc_1.csv"
    qrc_path_2 = "qrc_2.csv"
    qrc_path_3 = "qrc_3.csv"
    filter_qrc_path = "filter_qrc.csv"
    
    print(f"\nSTEP 1: For each document, ask LLM to write {num_q} questions answered in the document\n")

    generate_questions_for_documents(llm_q, num_q, doc_csv_schema, doc_path_0, doc_path_1)

    print("\nSTEP 2: Ask LLM to modify each question so that it makes a false assumption\n")

    infuse_questions_with_false_assumptions(doc_csv_schema, doc_path_1, doc_path_2)
    
    print("\nSTEP 3: Give LLM the document and the question and record LLM's response\n")

    generate_RAG_responses(llm_r, doc_csv_schema, doc_path_2, qrc_csv_schema, qrc_path_1)
    
    print("\nSTEP 4: Ask LLM to find the false assumption in each question\n")

    find_false_assumptions_in_questions(doc_csv_schema, doc_path_2, qrc_csv_schema, qrc_path_1, qrc_path_2)
    
    print("\nSTEP 5: Ask LLM if its initial response pointed out the false assumption\n")

    check_if_response_defused_confusion(doc_csv_schema, doc_path_2, qrc_csv_schema, qrc_path_2, qrc_path_3)
    
    print("\nSTEP 6: Compute performance metrics across all original and modified questions")

    filter_undefused_confusions_and_compute_metrics(qrc_csv_schema, qrc_path_3, filter_qrc_path)
    