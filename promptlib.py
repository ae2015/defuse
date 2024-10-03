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
        assert {"system", "user_reduce"}.issubset(prompts_raw.keys())
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
        assert {"system", "user_rag", "user_conf_rag", "user_def_check"}.issubset(prompts_raw.keys())
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
        # assert example["num_q"] == len(example_raw["conf_questions"])
        example["orig_questions"] = example_raw["orig_questions"]
        example["conf_questions"] = example_raw["conf_questions"]
        examples_of_questions[key] = example


def reduce_document(llm, document, num_fact, prompt_key):
    prompt = []
    if document_transforms[prompt_key]["system"]:
        prompt.append({
            "role" : "system",
            "content" : document_transforms[prompt_key]["system"]
        })
    prompt.append({
        "role" : "user",
        "content" : document_transforms[prompt_key]["user_reduce"].format(document = document, num_fact=num_fact)
    })
    reduce_doc = LLM.get(llm)(prompt)
    return reduce_doc


def modify_reduced_document(llm, document, reduce_doc, num_fact, prompt_key):
    if prompt_key in ["dt01", "dt02"]:
        return reduce_doc  
    doc_0 = reduce_doc
    doc_1 = suppress_facts(doc_0, lambda i: (i % 3 == 2))
    doc_2 = impute_facts(llm, doc_1, num_fact, prompt_key)
    doc_3 = suppress_facts(doc_2, lambda i: (i % 3 == 1))
    doc_4 = impute_facts(llm, doc_3, num_fact, prompt_key)
    doc_5 = suppress_facts(doc_4, lambda i: (i % 3 == 0))
    doc_6 = impute_facts(llm, doc_5, num_fact, prompt_key)

    doc_7 = suppress_facts(doc_6, lambda i: (i % 3 == 2))
    doc_8 = impute_facts(llm, doc_7, num_fact, prompt_key)
    doc_9 = suppress_facts(doc_8, lambda i: (i % 3 == 1))
    doc_10 = impute_facts(llm, doc_9, num_fact, prompt_key)
    doc_11 = suppress_facts(doc_10, lambda i: (i % 3 == 0))
    doc_12 = impute_facts(llm, doc_11, num_fact, prompt_key)
    modify_doc = doc_12
    remained_facts = remove_facts(llm, document, reduce_doc, modify_doc, num_fact, prompt_key)
    return remained_facts

def impute_facts(llm, missing_facts_doc, num_fact, prompt_key):
    prompt = []
    if document_transforms[prompt_key]["system"]:
        prompt.append({
            "role" : "system",
            "content" : document_transforms[prompt_key]["system"]
        })
    prompt.append({
        "role" : "user",
        "content" : document_transforms[prompt_key]["user_modify"].format(document = missing_facts_doc, num_fact=num_fact)
    })
    imputed_facts_doc = LLM.get(llm)(prompt)
    lines = imputed_facts_doc.splitlines()
    if "list of facts" in lines[0].lower():
        imputed_facts_doc = "\n".join(lines[1:])
    return imputed_facts_doc

def remove_facts(llm, document, ori_facts, hallucinated_facts, num_fact, prompt_key):
    prompt = []
    if document_transforms[prompt_key]["system"]:
        prompt.append({
            "role" : "system",
            "content" : document_transforms[prompt_key]["system"]
        })
    prompt.append({
        "role" : "user",
        "content" : document_transforms[prompt_key]["user_remove"].format(document = document, ori_facts = ori_facts, hallucinated_facts = hallucinated_facts, num_fact=num_fact)
    })
    remained_facts = LLM.get(llm)(prompt)
    # lines = imputed_facts_doc.splitlines()
    # if "list of facts" in lines[0].lower():
        # imputed_facts_doc = "\n".join(lines[1:])
    return remained_facts

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

def generate_questions(llm, document, num_q, prompt_key = "q-z-1"):
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

def confuse_questions_v2(llm, document, hallucinated_facts, prompt_key = "q-z-1"):
    '''
    Convert hallucinated facts into questions that can't be answered by the original document
    '''
    prompt = []
    if question_generation[prompt_key]["system"]:
        prompt.append({
            "role" : "system",
            "content" : question_generation[prompt_key]["system"]
        })
    prompt.append({
        "role" : "user",
        "content" : question_generation[prompt_key]["user_conf"].format(document = document, hallucinated_facts = hallucinated_facts)
    })
    # print("\n\n" + str(prompt) + "\n\n")
    raw_questions = LLM.get(llm)(prompt)
    questions = utils.parse_numbered_questions(raw_questions)
    return questions


def generate_response(llm, document, question, prompt_key = "r-z-1"):
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


def find_false_assumption(llm, document, question, prompt_key = "r-z-1"):
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
    
def find_false_assumption_v2(llm, document, question, n, prompt_key = "r-z-1"):
    exp_document = examples_of_questions["zpeng-sport-5-5"]["document"]
    exp_ori_questions, exp_ori_reasonings, exp_conf_questions, exp_conf_reasonings = [], [], [], []
    for t in examples_of_questions["zpeng-sport-5-5"]["conf_questions"]:
        exp_ori_questions.append(t["question"])
        exp_ori_reasonings.append(t["explanation"]+ " This answer is No.")
    for t in examples_of_questions["zpeng-sport-5-5"]["conf_questions"]:
        exp_conf_questions.append(t["question"])
        exp_conf_reasonings.append(t["explanation"]+ " This answer is Yes.")
    exp_questions = utils.enum_list(exp_ori_questions + exp_conf_questions)
    exp_reasonings = "\n\n" + utils.enum_list(exp_ori_reasonings + exp_conf_reasonings)
    prompt = []
    if rag_confusion_check[prompt_key]["system"]:
        prompt.append({
            "role" : "system",
            "content" : rag_confusion_check[prompt_key]["system"]
        })
    prompt.append({
        "role" : "user",
        "content" : rag_confusion_check[prompt_key]["user_conf_rag_example"].format(document = exp_document, question = exp_questions)
    })
    prompt.append({
        "role" : "assistant",
        "content" : exp_reasonings
    })
    prompt.append({
        "role" : "user",
        "content" : rag_confusion_check[prompt_key]["user_conf_rag_example"].format(document = document, question = question)
    })
    confusion = LLM.get(llm)(prompt, n=n)
    if n != 1:
        answers = []
        for conf in confusion:
            if (
                conf.lower().endswith("no") or
                conf.lower().endswith("answer: no") or
                conf.lower().endswith("the answer is: no") or
                conf.lower().endswith("the answer is \"no\"") or
                conf.lower().endswith("no.") or
                conf.lower().endswith("answer: no.") or
                conf.lower().endswith("the answer is: no.") or
                conf.lower().endswith("the answer is \"no.\"")
            ):
                answers.append("none")
            else:
                answers.append("yes")
        majority = max(answers, key = answers.count)
        return majority
    if (
            confusion.lower().endswith("no") or
            confusion.lower().endswith("answer: no") or
            confusion.lower().endswith("the answer is: no") or
            confusion.lower().endswith("the answer is \"no\"") or
            confusion.lower().endswith("no.") or
            confusion.lower().endswith("answer: no.") or
            confusion.lower().endswith("the answer is: no.") or
            confusion.lower().endswith("the answer is \"no.\"")
        ):
        return "none"
    else:
        return confusion


def check_response_for_defusion(llm, document, question, response, prompt_key = "r-z-1"):
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

def check_response_for_defusion_v2(llm, document, question, response, n, prompt_key = "r-z-1", shot = 2):
    prompt = []
    if rag_confusion_check[prompt_key]["system"]:
        prompt.append({
            "role" : "system",
            "content" : rag_confusion_check[prompt_key]["system"]
        })
    example_document = examples_of_questions["zpeng-sport-5-5"]["document"]
    example_questions, example_pos_answers, example_pos_reasonings, example_neg_answers, example_neg_reasonings = [], [], [], [], []
    for t in examples_of_questions["zpeng-sport-5-5"]["conf_questions"]:
        example_questions.append(t["question"])
        example_pos_answers.append(t["defuse"]["response"])
        example_pos_reasonings.append(t["defuse"]["reasoning"])
        example_neg_answers.append(t["rag"]["response"])
        example_neg_reasonings.append(t["rag"]["reasoning"])
    example_questions = utils.enum_list(example_questions[:shot]*2)
    example_answers = utils.enum_list(example_pos_answers[:shot]+example_neg_answers[:shot])
    example_reasonings = "\n\n" + utils.enum_list(example_pos_reasonings[:shot]+example_neg_reasonings[:shot])
    prompt.append({
        "role" : "user",
        "content" : rag_confusion_check[prompt_key]["user_rag"].format(document = example_document, question = example_questions)
    })
    prompt.append({
        "role" : "assistant",
        "content" : example_answers
    })
    prompt.append({
        "role" : "user",
        "content" : rag_confusion_check[prompt_key]["user_def_check"]
    })
    prompt.append({
        "role" : "assistant",
        "content" : example_reasonings
    })
    prompt.append({
        "role" : "user",
        "content" : rag_confusion_check[prompt_key]["user_rag"].format(document = document, question = question)
    })
    prompt.append({
        "role" : "assistant",
        "content" : response
    })
    prompt.append({
        "role" : "user",
        "content" : rag_confusion_check[prompt_key]["user_def_check"]
    })
    defusion = LLM.get(llm)(prompt, n=n)
    if n != 1:
        answers = []
        for defu in defusion:
            if (
                defu.lower().endswith("no") or
                defu.lower().endswith("answer: no") or
                defu.lower().endswith("the answer is: no") or
                defu.lower().endswith("the answer is \"no\"") or
                defu.lower().endswith("no.") or
                defu.lower().endswith("answer: no.") or
                defu.lower().endswith("the answer is: no.") or
                defu.lower().endswith("the answer is \"no.\"")
            ):
                answers.append("no")
                no_defu = defu
            elif (
                defu.lower().endswith("yes") or
                defu.lower().endswith("answer: yes") or
                defu.lower().endswith("the answer is: yes") or
                defu.lower().endswith("the answer is \"yes\"") or
                defu.lower().endswith("yes.") or
                defu.lower().endswith("answer: yes.") or
                defu.lower().endswith("the answer is: yes.") or
                defu.lower().endswith("the answer is \"yes.\"")
            ):
                answers.append("yes") 
                yes_defu = defu
            else:
                answers.append("unsure")
                unsure_defu = defu
        majority = max(answers, key = answers.count)
        if majority == "no":
            return no_defu, majority
        elif majority == "yes":
            return yes_defu, majority
        else:
            return unsure_defu, majority
    if (
            defusion.lower().endswith("no") or
            defusion.lower().endswith("answer: no") or
            defusion.lower().endswith("the answer is: no") or
            defusion.lower().endswith("the answer is \"no\"") or
            defusion.lower().endswith("no.") or
            defusion.lower().endswith("answer: no.") or
            defusion.lower().endswith("the answer is: no.") or
            defusion.lower().endswith("the answer is \"no.\"")
        ):
        is_defused = "no"
    elif (
            defusion.lower().endswith("yes") or
            defusion.lower().endswith("answer: yes") or
            defusion.lower().endswith("the answer is: yes") or
            defusion.lower().endswith("the answer is \"yes\"") or
            defusion.lower().endswith("yes.") or
            defusion.lower().endswith("answer: yes.") or
            defusion.lower().endswith("the answer is: yes.") or
            defusion.lower().endswith("the answer is \"yes.\"")
        ):
        is_defused = "yes"
    else:
        is_defused = "unsure"
    return defusion, is_defused
