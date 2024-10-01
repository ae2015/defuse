import pandas as pd
import os
import utils, promptlib
from tqdm import tqdm
from os.path import join
from dotenv import load_dotenv
load_dotenv()
print(os.getenv('OPENAI_API_KEY'))
def record_llm_and_prompts(llm, doc_prompt, schema, path_in, path_out):
    """
    Record LLM(s) and prompt(s) into the CSV table to use for generating confusing questions
    """
    df = utils.read_csv(path_in, "Read the input document table from CSV file")
    # Check that the correct columns are present in the CSV
    assert schema["document"] in set(df.columns)
    assert schema["LLM_q"] not in set(df.columns)
    assert schema["doc_prompt"] not in set(df.columns)
    df = df.reindex(columns = df.columns.tolist() + [schema["LLM_q"], schema["doc_prompt"]])
    df = df.astype({schema["LLM_q"]: str, schema["doc_prompt"]: str}, copy = False)
    for row_id, row in tqdm(df.copy().iterrows(), total = df.shape[0]):
        df.loc[row_id, schema["LLM_q"]] = llm
        df.loc[row_id, schema["doc_prompt"]] = doc_prompt
    utils.write_csv(df, path_out, "Write the document table with LLM names and prompt keys to CSV file")

def reduce_original_documents(schema, path_in, path_out):
    """
    Use LLM (or other means) to create a reduced version for each document
    """
    df = utils.read_csv(path_in, "Read the input document table from CSV file")
    # Check that the correct columns are present in the CSV
    assert schema["document"] in set(df.columns)
    assert schema["LLM_q"] in set(df.columns)
    assert schema["doc_prompt"] in set(df.columns)
    assert schema["reduce_doc"] not in set(df.columns)

    df = df.reindex(columns = df.columns.tolist() + [schema["reduce_doc"]])
    df = df.astype({schema["reduce_doc"]: str}, copy = False)
    print(f"Use LLM to create a reduced version for each document from column {schema['document']}")
    for row_id, row in tqdm(df.copy().iterrows(), total = df.shape[0]):
        llm = row[schema["LLM_q"]]
        prompt_key = row[schema["doc_prompt"]]
        document = utils.prepare_document(row[schema["document"]])
        reduce_doc = promptlib.reduce_document(llm, document, prompt_key)
        df.loc[row_id, schema["reduce_doc"]] = reduce_doc
    utils.write_csv(df, path_out, "Write the document table with the reduced versions to CSV file")


def modify_reduced_documents(schema, path_in, path_out):
    """
    Ask LLM to modify or impute information into each reduced document
    """
    df = utils.read_csv(path_in, "Read the reduced document table from CSV file")
    # Check that the correct columns are present in the CSV
    assert schema["LLM_q"] in set(df.columns)
    assert schema["doc_prompt"] in set(df.columns)
    assert schema["reduce_doc"] in set(df.columns)
    assert schema["modify_doc"] not in set(df.columns)

    df = df.reindex(columns = df.columns.tolist() + [schema["modify_doc"]])
    df = df.astype({schema["modify_doc"]: str}, copy = False)
    print(f"Use LLM to modify or impute information into each reduced document from column {schema['reduce_doc']}")
    for row_id, row in tqdm(df.copy().iterrows(), total = df.shape[0]):
        llm = row[schema["LLM_q"]]
        prompt_key = row[schema["doc_prompt"]]
        reduce_doc = utils.prepare_document(row[schema["reduce_doc"]])
        document = utils.prepare_document(row[schema["document"]])
        modify_doc = promptlib.modify_reduced_document(llm, document, reduce_doc, prompt_key)
        df.loc[row_id, schema["modify_doc"]] = modify_doc
    utils.write_csv(df, path_out, "Write the document table with the modified versions of reduced docs to CSV file")

def expand_modified_documents(schema, path_in, path_out):
    """
    Ask LLM to expand the modified/reduced version to the detailed document
    """
    df = utils.read_csv(path_in, "Read the modified/reduced document table from CSV file")
    # Check that the correct columns are present in the CSV
    assert schema["LLM_q"] in set(df.columns)
    assert schema["doc_prompt"] in set(df.columns)
    # assert schema["reduce_doc"] in set(df.columns)
    assert schema["modify_doc"] in set(df.columns)
    assert schema["expand_doc"] not in set(df.columns)

    df = df.reindex(columns = df.columns.tolist() + [schema["expand_doc"]])
    df = df.astype({schema["expand_doc"]: str}, copy = False)
    print(f"Use LLM to expand the modified/reduced version of the document from column {schema['modify_doc']}")
    for row_id, row in tqdm(df.copy().iterrows(), total = df.shape[0]):
        llm = row[schema["LLM_q"]]
        prompt_key = row[schema["doc_prompt"]]
        modify_doc = utils.prepare_document(row[schema["modify_doc"]])
        expand_doc = promptlib.expand_document(llm, modify_doc, prompt_key)
        df.loc[row_id, schema["expand_doc"]] = expand_doc
    utils.write_csv(df, path_out, "Write the document table with the expanded versions of reduced docs to CSV file")

def generate_questions_for_documents(num_q, schema, col_refs, path_in, path_out):
    """
    For each original document, ask LLM to write `num_q` questions answered in the document
    """
    doc_ref = col_refs[0]
    que_ref = col_refs[1]
    df = utils.read_csv(path_in, "Read the document table from CSV file")
    # Check that the correct columns are present in the CSV
    assert schema["LLM_q"] in set(df.columns)
    assert schema[doc_ref] in set(df.columns)
    assert schema[que_ref] not in set(df.columns)

    df = df.reindex(columns = df.columns.tolist() + [schema[que_ref]])
    df = df.astype({schema[que_ref]: str}, copy = False)
    print(f"Generate {num_q} questions for each document from column {schema[doc_ref]}")
    for row_id, row in tqdm(df.copy().iterrows(), total = df.shape[0]):
        llm = row[schema["LLM_q"]]
        document = utils.prepare_document(row[schema[doc_ref]])
        questions = promptlib.generate_questions(llm, document, num_q)
        df.loc[row_id, schema[que_ref]] = "\n".join([f"{i}. {q}" for i, q in enumerate(questions, start = 1)])
    utils.write_csv(df, path_out, "Write the document table with questions to CSV file")

def generate_questions_for_documents_v2(num_q, schema, col_refs, path_in, path_out):
    """
    Convert each hellucianted fact into a question which can only be answered by the hallucinated fact itself and can't be answered by the original document
    """
    doc_ref = col_refs[0]
    que_ref = col_refs[1]
    df = utils.read_csv(path_in, "Read the document table from CSV file")
    # Check that the correct columns are present in the CSV
    assert schema["LLM_q"] in set(df.columns)
    assert schema[doc_ref] in set(df.columns)
    assert schema[que_ref] not in set(df.columns)

    df = df.reindex(columns = df.columns.tolist() + [schema[que_ref]])
    df = df.astype({schema[que_ref]: str}, copy = False)
    print(f"Generate {num_q} questions for each document from column {schema[doc_ref]}")
    for row_id, row in tqdm(df.copy().iterrows(), total = df.shape[0]):
        llm = row[schema["LLM_q"]]
        document = utils.prepare_document(row[schema[doc_ref]])
        hallucinated_facts = utils.prepare_document(row[schema["modify_doc"]])
        questions = promptlib.confuse_questions_v2(llm, document, hallucinated_facts= hallucinated_facts)
        df.loc[row_id, schema[que_ref]] = "\n".join([f"{i}. {q}" for i, q in enumerate(questions, start = 1)])
    utils.write_csv(df, path_out, "Write the document table with questions to CSV file")


"""
def infuse_questions_with_false_assumptions(schema, path_in, path_out):
    df_in = utils.read_csv(path_in, "Read the document-and-questions table from CSV file")
    print("Modify each question by adding confusing (false) assumptions")
    rows_out = []
    for _, row in tqdm(df_in.iterrows(), total = df_in.shape[0]):
        doc_id = row[schema["doc_id"]]
        doc_source = row[schema["source"]]
        document = row[schema["document"]]
        llm = row[schema["LLM_q"]]
        orig_questions = utils.parse_numbered_questions(row[schema["orig_qs"]])
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
    """



def generate_RAG_responses(llm, doc_schema, doc_path, qr_schema, qr_path):
    df_in = utils.read_csv(doc_path, "Read the document-and-questions table from CSV file")
    print("Generate RAG response for each question, both original and confusing")
    rows_out = []
    for _, row in tqdm(df_in.iterrows(), total = df_in.shape[0]):
        doc_id = row[doc_schema["doc_id"]]
        document = row[doc_schema["document"]]
        orig_questions = utils.parse_numbered_questions(row[doc_schema["orig_qs"]])
        conf_questions = utils.parse_numbered_questions(row[doc_schema["conf_qs"]])
        qs = (
            [(q_id, "no" , q) for q_id, q in enumerate(orig_questions, start = 1)] +
            [(q_id, "yes", q) for q_id, q in enumerate(conf_questions, start = 1)]
        )
        for q_id, is_conf, q in qs:
            response_q = promptlib.generate_response(llm, document, q)
            row_out = {
                qr_schema["doc_id"] : doc_id,
                qr_schema["q_id"] : q_id,
                qr_schema["is_conf"] : is_conf,
                qr_schema["question"] : q,
                qr_schema["LLM_r"] : llm,
                qr_schema["response"] : response_q
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
            defusion, is_defused = promptlib.check_response_for_defusion(llm, document, question, response)
        row_out = dict(row)
        row_out[qr_schema["defusion"]] = defusion
        row_out[qr_schema["is_defused"]] = is_defused
        rows_out.append(row_out)
    df_out = pd.DataFrame.from_dict(rows_out, dtype = str)
    utils.write_csv(df_out, qr_path_out, "Write the question-response table to CSV file")

def filter_undefused_confusions_and_compute_metrics(qr_schema, qr_path, filter_path, metrics_path):
    df_qr = utils.read_csv(qr_path, "Read the question-response table from CSV file")
    f = open(metrics_path, "w")
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
    f.write("Original (non-confusing) questions:")
    f.write(f"\n    Total questions = {num_orig_questions}")
    f.write(f"\n    With confusion detected = {num_orig_questions_with_confusion_detected}")
    f.write(f"\n    With confusion detected and defused = {num_orig_questions_with_conf_detected_and_defused}")
    f.write("\n Confusing questions:")
    num_conf_questions_with_conf_detected_but_undefused = \
        num_conf_questions_with_confusion_detected - num_conf_questions_with_conf_detected_and_defused
    f.write(f"\n    Total questions = {num_conf_questions}")
    f.write(f"\n    With confusion detected = {num_conf_questions_with_confusion_detected}")
    f.write(f"\n    With confusion detected and defused = {num_conf_questions_with_conf_detected_and_defused}")
    f.write(f"\n    With confusion detected, but not defused = {num_conf_questions_with_conf_detected_but_undefused}")

    df_filter = pd.DataFrame.from_dict(filter_rows, dtype = str)
    utils.write_csv(df_filter, filter_path, "Write the filtered question-response table to CSV file")
    f.close()



if __name__ == "__main__":

    doc_csv_schema = {
        "doc_id" : "doc_id",           # Column with a unique document ID
        "source" : "source",           # Column with document source (e.g. URL)
        "document" : "document",       # Column with the text of the original document
        "LLM_q" : "LLM_q",             # Column with name of the LLM used to generate confusing questions
        "doc_prompt" : "doc_prompt",   # Column with JSON key of the prompt used to transform document
        "reduce_doc" : "reduce_doc",   # Column with the text of the reduced / simplified document
        "modify_doc" : "modify_doc",   # Column with the text of the modified / confused document
        "expand_doc" : "expand_doc",   # Column with the text of the expanded / reconstructed document
        "orig_qs" : "orig_questions",  # Column with the questions to original document (non-confusing)
        "conf_qs" : "conf_questions"   # Column with the questions to expanded document (confusing)
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

    llm_q = "gpt-4o-mini"  # LLM for generating questions (stronger)
    llm_r = "gpt-3.5"  # LLM for generating responses (weaker)

    doc_prompt = "dt03"

    num_q_orig =  5  # Number of questions per original document
    num_q_conf = 10  # Number of questions per expanded document
    # topics = [
    #     'sport', 'business', 'science', 'finance', 'food',
    #     'politics', 'economics', 'travel', 'entertainment',
    #     'music', 'news'
    # ]
    topics = [
        'sport'
    ]
    for topic in topics:
        data_folder = f"data/processed/News1k2024-300/{topic}/20"
        exp_folder = f"data/experiments/llmq-{llm_q}/llmr-{llm_r}/docp-{doc_prompt}/20-toy-update/{topic}"
        os.makedirs(exp_folder, exist_ok = True)
        doc_files = {
            "in" : "docs_in.csv",
            "out" : "docs_out.csv",
            0 : "docs_0.csv",
            1 : "docs_1.csv",
            2 : "docs_2.csv",
            3 : "docs_3.csv",
            4 : "docs_4.csv"
        }
        qrc_files = {
            "out" : "qrc_out.csv",
            "filter" : "qrc_filter.csv",
            1 : "qrc_1.csv",
            2 : "qrc_2.csv"
        }

        doc_paths = {k : join(data_folder, v) if k == "in" else join(exp_folder, v)  for k, v in doc_files.items()}
        qrc_paths = {k : join(exp_folder, v) for k, v in qrc_files.items()}
        metric_path = join(exp_folder, "metrics.txt")
        promptlib.read_prompts("prompts")

        print(f"\nSTEP 0: Record LLM(s) and prompt(s) to use for generating confusing questions\n")

        record_llm_and_prompts(llm_q, doc_prompt, doc_csv_schema, doc_paths["in"], doc_paths[0])
        
        print(f"\nSTEP 1: Use LLM (or other means) to create a reduced version for each document\n")
        
        reduce_original_documents(doc_csv_schema, doc_paths[0], doc_paths[1])

        print(f"\nSTEP 2: Ask LLM to modify or impute information into each reduced document\n")

        modify_reduced_documents(doc_csv_schema, doc_paths[1], doc_paths[2])

        # print(f"\nSTEP 3: Ask LLM to expand the modified/reduced version to the detailed document\n")

        # expand_modified_documents(doc_csv_schema, doc_paths[2], doc_paths[3])

        print(f"\nSTEP 3: For each original document, ask LLM to write " +
            f"{num_q_orig} questions answered in the document\n")

        generate_questions_for_documents(num_q_orig, doc_csv_schema, ["document", "orig_qs"],
                                        doc_paths[2], doc_paths[3])

        print(f"\nSTEP 4: For each expanded document, ask LLM to write " +
            f"{num_q_conf} questions answered in the document\n")

        generate_questions_for_documents_v2(num_q_conf, doc_csv_schema, ["document", "conf_qs"],
                                        doc_paths[3], doc_paths["out"])

        print("\nSTEP 5: Give LLM the document and the question and record LLM's response\n")

        generate_RAG_responses(llm_r, doc_csv_schema, doc_paths["out"], qrc_csv_schema, qrc_paths[1])
        
        
        print("\nSTEP 6: Ask LLM to find the false assumption in each question\n")

        find_false_assumptions_in_questions(doc_csv_schema, doc_paths["out"],
                                            qrc_csv_schema, qrc_paths[1], qrc_paths[2])
        
        print("\nSTEP 7: Ask LLM if its initial response pointed out the false assumption\n")

        check_if_response_defused_confusion(doc_csv_schema, doc_paths["out"],
                                            qrc_csv_schema, qrc_paths[2], qrc_paths["out"])
        
        print("\nSTEP 8: Compute performance metrics across all original and modified questions")

        filter_undefused_confusions_and_compute_metrics(qrc_csv_schema, qrc_paths["out"], qrc_paths["filter"], metric_path)
    