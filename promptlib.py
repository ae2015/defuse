from llmlib import LLM
import utils
import os, re, json

document_transforms = None
question_generation = None
rag_confusion_check = None
examples_of_questions = None

def read_prompts(folder):

    global document_transforms
    with open(os.path.join(folder, "document-transforms.json"), "r") as f:
        document_transforms_raw = json.load(f)
    document_transforms = {}
    for key, prompts_raw in document_transforms_raw.items():
        assert {"system", "user_reduce", "user_expand"}.issubset(prompts_raw.keys())
        prompts = {}
        for p_type in prompts_raw.keys():
            if isinstance(prompts_raw[p_type], str) or prompts_raw[p_type] is None:
                prompts[p_type] = prompts_raw[p_type]
            else:
                prompts[p_type] = "".join(prompts_raw[p_type])
        document_transforms[key] = prompts

    global question_generation
    with open(os.path.join(folder, "question-generation.json"), "r") as f:
        question_generation_raw = json.load(f)
    question_generation = {}
    for key, prompts_raw in question_generation_raw.items():
        assert {"system", "user_orig", "user_conf"}.issubset(prompts_raw.keys())
        prompts = {}
        for p_type in prompts_raw.keys():
            if isinstance(prompts_raw[p_type], str) or prompts_raw[p_type] is None:
                prompts[p_type] = prompts_raw[p_type]
            else:
                prompts[p_type] = "".join(prompts_raw[p_type])
        question_generation[key] = prompts

    global rag_confusion_check
    with open(os.path.join(folder, "rag-confusion-check.json"), "r") as f:
        rag_confusion_check_raw = json.load(f)
    rag_confusion_check = {}
    for key, prompts_raw in rag_confusion_check_raw.items():
        assert {"system", "user_rag", "user_conf_rag", "user_conf_check", "user_def_check"}.issubset(prompts_raw.keys())
        prompts = {}
        for p_type in prompts_raw.keys():
            if isinstance(prompts_raw[p_type], str) or prompts_raw[p_type] is None:
                prompts[p_type] = prompts_raw[p_type]
            else:
                prompts[p_type] = "".join(prompts_raw[p_type])
        rag_confusion_check[key] = prompts

    global examples_of_questions
    with open(os.path.join(folder, "examples-of-questions.json"), "r") as f:
        examples_of_questions_raw = json.load(f)
    examples_of_questions = {}
    for key, example_raw in examples_of_questions_raw.items():
        assert {"document", "source", "orig_questions", "conf_questions"}.issubset(example_raw.keys())
        example = {}
        if isinstance(example_raw["document"], str):
            example["document"] = example_raw["document"]
        else:
            example["document"] = "".join(example_raw["document"])
        assert isinstance(example_raw["orig_questions"], list)
        assert isinstance(example_raw["conf_questions"], list)
        example["num_q"] = len(example_raw["orig_questions"])
        assert example["num_q"] == len(example_raw["conf_questions"])
        example["orig_questions"] = example_raw["orig_questions"]
        example["conf_questions"] = example_raw["conf_questions"]
        examples_of_questions[key] = example


def reduce_document(llm, document, prompt_key):
    prompt = []
    if document_transforms[prompt_key]["system"]:
        prompt.append({
            "role" : "system",
            "content" : document_transforms[prompt_key]["system"]
        })
    prompt.append({
        "role" : "user",
        "content" : document_transforms[prompt_key]["user_reduce"].format(document = document)
    })
    reduce_doc = LLM.get(llm)(prompt)
    return reduce_doc


def modify_reduced_document(llm, reduce_doc, prompt_key):
    if prompt_key in ["dt01", "dt02"]:
        return reduce_doc  
    doc_0 = reduce_doc
    doc_1 = suppress_facts(doc_0, lambda i: (i % 3 == 2))
    doc_2 = impute_facts(llm, doc_1, prompt_key)
    doc_3 = suppress_facts(doc_2, lambda i: (i % 3 == 1))
    doc_4 = impute_facts(llm, doc_3, prompt_key)
    doc_5 = suppress_facts(doc_4, lambda i: (i % 3 == 0))
    doc_6 = impute_facts(llm, doc_5, prompt_key)
    modify_doc = doc_6
    return modify_doc

def impute_facts(llm, missing_facts_doc, prompt_key):
    prompt = []
    if document_transforms[prompt_key]["system"]:
        prompt.append({
            "role" : "system",
            "content" : document_transforms[prompt_key]["system"]
        })
    prompt.append({
        "role" : "user",
        "content" : document_transforms[prompt_key]["user_modify"].format(document = missing_facts_doc)
    })
    imputed_facts_doc = LLM.get(llm)(prompt)
    lines = imputed_facts_doc.splitlines()
    if "list of facts" in lines[0].lower():
        imputed_facts_doc = "\n".join(lines[1:])
    return imputed_facts_doc

def suppress_facts(text, suppress):
    raw_lines = text.splitlines()
    lines = [line.strip() for line in raw_lines]
    facts = []
    for line in lines:
        if len(line) > 0:
            x = re.search(r"^\d+[:\.]\s+", line)
            if x:
                facts.append(line[x.span()[1]:])
            else:
                x = re.search(r"^[:\.\-\*\+]\s+", line)
                if x:
                    facts.append(line[x.span()[1]:])
                else:
                    facts.append(line)
    # print(f"\n\n{utils.enum_list(facts)}\n\n")
    for i in range(len(facts)):
        if suppress(i):
            facts[i] = "(missing)"
    # print(f"\n\n{utils.enum_list(facts)}\n\n")
    return utils.enum_list(facts)

def expand_document(llm, reduce_doc, prompt_key):
    prompt = []
    if document_transforms[prompt_key]["system"]:
        prompt.append({
            "role" : "system",
            "content" : document_transforms[prompt_key]["system"]
        })
    prompt.append({
        "role" : "user",
        "content" : document_transforms[prompt_key]["user_expand"].format(document = reduce_doc)
    })
    expand_doc = LLM.get(llm)(prompt)
    return expand_doc



def generate_questions(llm, document, num_q, prompt_key = "q01"):
    example_keys = ["Weywot-1", "ElDorado-1"]
    prompt = []
    if question_generation[prompt_key]["system"]:
        prompt.append({
            "role" : "system",
            "content" : question_generation[prompt_key]["system"]
        })
    for key in example_keys:
        ex_document = examples_of_questions[key]["document"]
        ex_num_q = examples_of_questions[key]["num_q"]
        prompt.append({
            "role" : "user",
            "content" : question_generation[prompt_key]["user_orig"].format(num_q = ex_num_q, document = ex_document)
        })
        prompt.append({
            "role" : "assistant",
            "content" : utils.enum_list(examples_of_questions[key]["orig_questions"])
        })
    prompt.append({
        "role" : "user",
        "content" : question_generation[prompt_key]["user_orig"].format(num_q = num_q, document = document)
    })
    # print("\n\n" + str(prompt) + "\n\n")
    raw_questions = LLM.get(llm)(prompt)
    questions = utils.parse_numbered_questions(raw_questions)
    return questions


def confuse_questions(llm, document, questions, prompt_key = "q01"):
    example_keys = ["Weywot-1", "ElDorado-1"]
    prompt = []
    if question_generation[prompt_key]["system"]:
        prompt.append({
            "role" : "system",
            "content" : question_generation[prompt_key]["system"]
        })
    for key in example_keys:
        ex_document = examples_of_questions[key]["document"]
        ex_num_q = examples_of_questions[key]["num_q"]
        prompt.append({
            "role" : "user",
            "content" : question_generation[prompt_key]["user_orig"].format(num_q = ex_num_q, document = ex_document)
        })
        prompt.append({
            "role" : "assistant",
            "content" : utils.enum_list(examples_of_questions[key]["orig_questions"])
        })
        prompt.append({
            "role" : "user",
            "content" : question_generation[prompt_key]["user_conf"].format(num_q = ex_num_q, document = ex_document)
        })
        prompt.append({
            "role" : "assistant",
            "content" : utils.enum_list(examples_of_questions[key]["conf_questions"])
        })
    prompt.append({
        "role" : "user",
        "content" : question_generation[prompt_key]["user_orig"].format(num_q = len(questions), document = document)
    })
    prompt.append({
        "role" : "assistant",
        "content" : utils.enum_list(questions)
    })
    prompt.append({
        "role" : "user",
        "content" : question_generation[prompt_key]["user_conf"].format(num_q = len(questions), document = document)
    })
    # print("\n\n" + str(prompt) + "\n\n")
    raw_questions = LLM.get(llm)(prompt)
    questions = utils.parse_numbered_questions(raw_questions)
    return questions


def generate_response(llm, document, question, prompt_key = "r02"):
    prompt = []
    if rag_confusion_check[prompt_key]["system"]:
        prompt.append({
            "role" : "system",
            "content" : rag_confusion_check[prompt_key]["system"]
        })
    prompt.append({
        "role" : "user",
        "content" : rag_confusion_check[prompt_key]["user_rag"].format(document = document, question = question)
    })
    response = LLM.get(llm)(prompt)
    return response


def find_false_assumption(llm, document, question, prompt_key = "r02"):
    prompt = []
    if rag_confusion_check[prompt_key]["system"]:
        prompt.append({
            "role" : "system",
            "content" : rag_confusion_check[prompt_key]["system"]
        })
    prompt.append({
        "role" : "user",
        "content" : rag_confusion_check[prompt_key]["user_conf_rag"].format(document = document, question = question)
    })
    confusion = LLM.get(llm)(prompt)
    if (
            confusion.lower().startswith("no") or
            confusion.lower().startswith("answer: no") or
            confusion.lower().startswith("the answer is: no") or
            confusion.lower().startswith("the answer is \"no\"") or
            confusion.lower().startswith("The answer to the question is: no") or
            confusion.lower().startswith("The answer to the question is \"no\"") or
            "the question does not contain a confusing part" in confusion.lower()
        ):
        return "none"
    else:
        return confusion


def check_response_for_defusion(llm, document, question, response, confusion, prompt_key = "r02"):
    prompt = []
    if rag_confusion_check[prompt_key]["system"]:
        prompt.append({
            "role" : "system",
            "content" : rag_confusion_check[prompt_key]["system"]
        })
    prompt.append({
        "role" : "user",
        "content" : rag_confusion_check[prompt_key]["user_rag"].format(document = document, question = question)
    })
    prompt.append({
        "role" : "assistant",
        "content" : response
    })
    """
    prompt.append({
        "role" : "user",
        "content" : rag_confusion_check[prompt_key]["user_conf_check"]
    })
    prompt.append({
        "role" : "assistant",
        "content" : confusion
    })
    """
    prompt.append({
        "role" : "user",
        "content" : rag_confusion_check[prompt_key]["user_def_check"]
    })
    defusion = LLM.get(llm)(prompt)
    if (
            defusion.lower().startswith("no") or
            defusion.lower().startswith("answer: no") or
            defusion.lower().startswith("the answer is: no") or
            defusion.lower().startswith("the answer is \"no\"")
        ):
        is_defused = "no"
    elif (
            defusion.lower().startswith("yes") or
            defusion.lower().startswith("answer: yes") or
            defusion.lower().startswith("the answer is: yes") or
            defusion.lower().startswith("the answer is \"yes\"")
        ):
        is_defused = "yes"
    else:
        is_defused = "unsure"
    return defusion, is_defused
