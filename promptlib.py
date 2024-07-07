from llmlib import LLM
import utils

def generate_questions(llm, document, num_q):
    prompt = (
        f"Read the document and write a numbered list of {num_q} best " +
         "questions that are answered in the document.\n\n" +
        f"Document:\n\n{document}\n\n" + "Questions:\n"
    )
    raw_questions = LLM.get(llm)(prompt)
    questions = utils.parse_numbered_list(raw_questions)
    return questions

def confuse_questions(llm, document, questions):
    prompt_for_confusion = (
            "Now, slightly modify each question in the list " + 
            "by replacing one or more words with other relevant " +
            "words from the document, " +
            "so that each new question makes a false assumption " +
            "and has no good answer."
        )
    prompt = [
        (
            "Read the document and write a numbered list of 10 best " +
            "questions that are answered in the document.\n\n" +
            "Document:\n\n" +
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
            "Apartments today!\n\n" + "Questions:"
        ),
        (
            "1. What types of apartment units are available at Weywot Apartments?\n" +
            "2. How much is the monthly rent for a 1-bedroom apartment?\n" +
            "3. What is the square footage of the 2-bedroom units?\n" +
            "4. What amenities are included within the individual apartments?\n" +
            "5. Are there any community features available at Weywot Apartments?\n" +
            "6. Is parking available for residents at Weywot Apartments?\n" +
            "7. What security measures are in place at Weywot Apartments?\n" +
            "8. How close are Weywot Apartments to local shops and public transportation?\n" +
            "9. What is the monthly rent for a 2-bedroom apartment?\n" +
            "10. What size is a 1-bedroom unit in terms of square footage?\n"
        ),
        prompt_for_confusion,
        (
            "1. What types of apartment units are on sale at Weywot Apartments?\n" +
            "2. How much is the weekly rent for a 1-bedroom apartment?\n" +
            "3. What is the cubic footage of the 2-bedroom units?\n" +
            "4. What necessities are excluded within the individual apartments?\n" +
            "5. Are there any accessibility features available at Weywot Apartments?\n" +
            "6. How much is the parking for residents at Weywot Apartments?\n" +
            "7. What security measures are lacking at Weywot Apartments?\n" +
            "8. How close are Weywot Apartments to local schools and subway stations?\n" +
            "9. What is the monthly mortgage for a 2-bedroom apartment?\n" +
            "10. What size is a 1-bedroom kitchen in terms of square footage?\n"
        ),
        (
            f"Read the document and write a numbered list of {len(questions)} best " +
             "questions that are answered in the document.\n\n" +
            f"Document:\n\n{document}\n\n" + "Questions:"
        ),
        "\n".join([f"{i}. {q}" for i, q in enumerate(questions, start = 1)]),
        prompt_for_confusion
    ]
    raw_questions = LLM.get(llm)(prompt)
    questions = utils.parse_numbered_list(raw_questions)
    return questions

def generate_response(llm, document, question):
    prompt = (
        f"Read the document and answer the question based on the document.\n\n" +
        f"Document:\n\n{document}\n\nQuestion:\n\n{question}\n\nAnswer:"
    )
    response = LLM.get(llm)(prompt)
    return response

def find_false_assumption(llm, document, question):
    prompt = (
        f"Read the document and the question, then check whether the question contains " +
         "a false assumption.\n\n" +
        f"Document:\n\n{document}\n\nQuestion:\n\n{question}\n\n" +
         "Does the question contain a false assumption? Answer 'Yes' or 'No', then " +
         "provide the false assumption if the question has it:"
    )
    confusion = LLM.get(llm)(prompt)
    if confusion[:2].lower() == "no":
        return "none"
    else:
        return confusion

def check_response_for_defusion(llm, document, question, response, confusion):
    prompt = [
        (
            f"Read the document and answer the question based on the document.\n\n" +
            f"Document:\n\n{document}\n\nQuestion:\n\n{question}\n\nAnswer:"
        ),
        response,
        (
            "Does the question contain a false assumption? Answer 'Yes' or 'No', then " +
            "provide the false assumption if the question has it:"
        ),
        confusion,
        (
            "Now read the answer you gave to the question. Did your answer point out " +
            "the false assumption in the question? Answer 'Yes' or 'No', then explain:"
        )
    ]
    defusion = LLM.get(llm)(prompt)
    if defusion[:2].lower() == "no":
        is_defused = "no"
    elif defusion[:3].lower() == "yes":
        is_defused = "yes"
    else:
        is_defused = "unsure"
    return defusion, is_defused
