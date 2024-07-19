import pandas as pd
import re

def read_csv(path, comment):
    print(comment + ":\n    " + path)
    df = pd.read_csv(path, dtype = str, na_filter = False)
    print("    Rows: " + str(len(df)) + ",  Cols: " + str(len(df.columns)))
    print("    " + str(df.columns))
    return df

def write_csv(df, path, comment):
    print(comment + ":\n    " + path)
    print("    Rows: " + str(len(df)) + ",  Cols: " + str(len(df.columns)))
    print("    " + str(df.columns))
    df.to_csv(path, index = False)


    # text_output = text_output.strip()
    # if (len(text_output) >= 2 and
    #         text_output[0] == text_output[-1] and text_output[0] in ["'", '"']):
    #     text_output = text_output[1 : -1].strip()
       
def prepare_document(raw_document):
    document = re.sub(r"\n\s*\n", "\n", raw_document)  # Remove excessive empty lines
    return document

def enum_list(questions):
    return "\n".join([f"{i}. {q}" for i, q in enumerate(questions, start = 1)])

def parse_numbered_questions(text): # , min_number_of_items = 2):
    lines = text.splitlines()
    questions = []
    chunks = [] # One question could span multiple lines
    def add_question_from_chunks():
        nonlocal chunks, questions
        if chunks:
            question = (" ".join(chunks)).strip()
            if question[-1] == '?':
                questions.append(question)
            chunks = []
    for raw_line in lines:
        line = raw_line.strip()
        if len(line) > 0:
            x = re.search(r"^\d+[:\.]\s+", line)
            if x:  # The line starts a new question
                add_question_from_chunks()
                line = line[x.span()[1]:]
            chunks.append(line)
            if line[-1] == '?':
                add_question_from_chunks()
    add_question_from_chunks()
    # print(questions)
    return questions


