from llmlib import LLM
import utils

prompt_to_make_confusion = (
    "Now, slightly modify each question in the list " + 
    "by replacing one or more words with other relevant " +
    "words from the document, " +
    "so that each new question makes a false assumption " +
    "and has no good answer:"
)
prompt_to_check_confusion = (
    "Does the question contain a false assumption? Answer 'Yes' or 'No', then " +
    "provide the false assumption if the question has it:"
)
prompt_to_check_defusion = (
    "Now read the answer you gave to the question. Did your answer point out " +
    "the false assumption in the question? Answer 'Yes' or 'No', then explain:"
)

def prompt_for_questions(document, num_q):
    return (
        f"Read the document and write a numbered list of {num_q} best " +
         "questions that are answered in the document.\n\n" +
        f"Document:\n\n{document}\n\nQuestions:"
    )

def prompt_for_RAG(document, question):
    return (
        "Read the document and answer the question based on the document.\n\n" +
        f"Document:\n\n{document}\n\nQuestion:\n\n{question}\n\nAnswer:"
    )

def prompt_for_RAG_confusion_check(document, question):
    return (
        f"Read the document and the question, then check whether the question contains " +
         "a false assumption.\n\n" +
        f"Document:\n\n{document}\n\nQuestion:\n\n{question}\n\n" +
        prompt_to_check_confusion
    )

examples_of_making_confusion = [
    {
        "document" : (
            "Weywot Apartments is a luxury apartment community offering a unique living experience, " +
            "nestled in the heart of our thriving neighborhood. Our well-maintained properties come " +
            "equipped with modern amenities and a cozy atmosphere perfect for any lifestyle. We offer " +
            "spacious 1-bedroom and 2-bedroom apartment homes for rent at affordable prices. Our " +
            "apartments are designed with your comfort and convenience in mind, featuring spacious " +
            "living quarters, modern kitchens with solid wood cabinets and Corian countertops, as well " +
            "as private patios or decks for you to enjoy. Our 1-bedroom units feature 900 sq.ft. " +
            "living area and 1 full bath, offered for $2,000 per month. For those who need more space, " +
            "our 2-bedroom units offer 1200 sq.ft. living area and 1 full bath, available for $3,000 " +
            "per month. All of our apartments come equipped with in-home washers and dryers, heating " +
            "and air conditioning, as well as high-speed in-home and common area WiFi. A covered parking " +
            "spot is provided for each apartment. In addition to our individual apartment amenities, " +
            "Weywot Apartments offers a variety of community features including a pool and a fitness center. " +
            "Our apartments are located in a secure, gated community with controlled access, ensuring your " +
            "safety and peace of mind. Enjoy the convenience of being just steps away from local shops, " +
            "restaurants, and public transportation. Our team is committed to providing top-notch customer " +
            "service and addressing all your needs promptly. Experience the tranquility of living in Weywot " +
            "Apartments today!"
        ),
        "orig_questions" : [
            "What types of apartment units are available at Weywot Apartments?",
            "How much is the monthly rent for a 1-bedroom apartment?",
            "What is the square footage of the 2-bedroom units?",
            "What amenities are included within the individual apartments?",
            "Are there any community features available at Weywot Apartments?",
            "Is parking available for residents at Weywot Apartments?",
            "What security measures are in place at Weywot Apartments?",
            "How close are Weywot Apartments to local shops and public transportation?",
            "What is the monthly rent for a 2-bedroom apartment?",
            "What size is a 1-bedroom unit in terms of square footage?"
        ],
        "conf_questions" : [
            "What types of apartment units are on sale at Weywot Apartments?",
            "How much is the weekly rent for a 1-bedroom apartment?",
            "What is the perimeter footage of the 2-bedroom units?",
            "What necessities are excluded within the individual apartments?",
            "Are there any accessibility features available at Weywot Apartments?",
            "How much is the parking fee for residents at Weywot Apartments?",
            "What security measures are in plan at Weywot Apartments?",
            "How close are Weywot Apartments to local schools and subway stations?",
            "What is the monthly tax for a 2-bedroom apartment?",
            "What size is a 1-bedroom kitchen in terms of square footage?"
        ]
    },
    {
        "document" : (
            "The El Dorado County Library immediately put its 3-D printers to work creating masks and face shields " +
            "for healthcare workers. By early April 2020, they had formed a partnership with a local pharmaceutical " +
            "startup to help produce and distribute thousands of face shields to local medical personnel and " +
            "frontline workers. By October, the library was also partnering with the El Dorado County Registrar of " +
            "Voters to provide more face shields to poll workers, as well as hosting a voting center and drive-up " +
            "ballot collection boxes at branches countywide.\n" +
            "The library distributes free food, diapers, and other essentials in partnership with the Placer Food " +
            "Bank, El Dorado Community Foundation, and First 5 El Dorado Commission. Library staff also help " +
            "community members register for vaccine appointments online. Many in the county do not have computers " +
            "or access to the internet. The library received 200 calls and had dozens of people waiting at the " +
            "library doors in the first three hours of offering this service."
        ),
        "orig_questions" : [
            "How did the El Dorado County Library utilize its 3-D printers during the COVID-19 pandemic?",
            "With which organization did the library form a partnership to produce and distribute face shields?",
            "When did the library begin its partnership with the local startup for face shield production?",
            "What additional partnership did the library establish by October 2020?",
            "What services did the library provide in collaboration with the El Dorado County Registrar of Voters?",
            "Which organizations does the library partner with to distribute free food, diapers, and other essentials?",
            "How does the library assist community members with vaccine appointments?",
            "What challenges do many community members in El Dorado County face in accessing the internet?",
            "How many calls did the library receive in the first 3 hours of offering vaccine appointment assistance?",
            "What immediate impact did the library's vaccine appointment registration service have on the community?"
        ],
        "conf_questions" : [
            "How did the El Dorado County Library utilize its healthcare workers during the COVID-19 pandemic?",
            "With which organization did the library sign a contract to produce and distribute face shields?",
            "When did the library staff begin their employment with the local startup for face shield production?",
            "What additional organization did the library establish by October 2020?",
            "What products did the library sell in collaboration with the El Dorado County Registrar of Voters?",
            "Which organizations does the library supply with free food, diapers, and other essentials?",
            "How does the library convince community members to get vaccine appointments?",
            "What challenges does the majority of the community members in El Dorado County face in accessing the internet?",
            "How many police calls did the library receive in the first 3 hours of offering vaccine appointment assistance?",
            "What immediate effect did the library's vaccine appointment registration service have on the spread of COVID-19?"
        ]
    }
]


def generate_questions(llm, document, num_q):
    prompt = prompt_for_questions(document, num_q)
    raw_questions = LLM.get(llm)(prompt)
    questions = utils.parse_numbered_list(raw_questions)
    return questions

def confuse_questions(llm, document, questions):
    prompt = [
        prompt_for_questions(
            examples_of_making_confusion[0]["document"],
            len(examples_of_making_confusion[0]["orig_questions"])
        ),
        utils.enum_list(examples_of_making_confusion[0]["orig_questions"]),
        prompt_to_make_confusion,
        utils.enum_list(examples_of_making_confusion[0]["conf_questions"]),
        prompt_for_questions(
            examples_of_making_confusion[1]["document"],
            len(examples_of_making_confusion[1]["orig_questions"])
        ),
        utils.enum_list(examples_of_making_confusion[1]["orig_questions"]),
        prompt_to_make_confusion,
        utils.enum_list(examples_of_making_confusion[1]["conf_questions"]),
        prompt_for_questions(document, len(questions)),
        utils.enum_list(questions),
        prompt_to_make_confusion
    ]
    raw_questions = LLM.get(llm)(prompt)
    questions = utils.parse_numbered_list(raw_questions)
    return questions

def generate_response(llm, document, question):
    prompt = prompt_for_RAG(document, question)
    response = LLM.get(llm)(prompt)
    return response

def find_false_assumption(llm, document, question):
    prompt = prompt_for_RAG_confusion_check(document, question)
    confusion = LLM.get(llm)(prompt)
    if confusion[:2].lower() == "no":
        return "none"
    else:
        return confusion

def check_response_for_defusion(llm, document, question, response, confusion):
    prompt = [
        prompt_for_RAG(document, question),
        response,
        prompt_to_check_confusion,
        confusion,
        prompt_to_check_defusion
    ]
    defusion = LLM.get(llm)(prompt)
    if defusion[:2].lower() == "no":
        is_defused = "no"
    elif defusion[:3].lower() == "yes":
        is_defused = "yes"
    else:
        is_defused = "unsure"
    return defusion, is_defused
