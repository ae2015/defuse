import requests
import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
from nltk import sent_tokenize


def split_sentences(text):
    return sent_tokenize(text)

def extract_triples_compact_ie(text: str):
    """Uses the CompactIE API to extract triples, running locally"""
    request = {"sentences": [s.strip() for s in split_sentences(text)]}
    result = requests.post("http://0.0.0.0:39881/api", json=request).json()
    return [(a["subject"], a["relation"], a["object"]) for a in result]

def main():
    texts = [
        "Peugeot deal boosts Mitsubishi Struggling Japanese car maker Mitsubishi Motors has struck a deal to supply French car maker Peugeot with 30,000 sports utility vehicles (SUV).", "The two firms signed a Memorandum of Understanding, and say they expect to seal a final agreement by Spring 2005.",
        "The alliance comes as a badly-needed boost for loss-making Mitsubishi, after several profit warnings and poor sales.",
        "The SUVs will be built in Japan using Peugeot's diesel engines and sold mainly in the European market. Falling sales have left Mitsubishi Motors with underused capacity, and the production deal with Peugeot gives it a chance to utilise some of it."]
        # In January, Mitsubishi Motors issued its third profits warning in nine months, and cut its sales forecasts for the year to March 2005. Its sales have slid 41% in the past year, catalysed by the revelation that the company had systematically been hiding records of faults and then secretly repairing vehicles. As a result, the Japanese car maker has sought a series of financial bailouts. Last month it said it was looking for a further 540bn yen ($5.2bn; 2.77bn pounds) in fresh financial backing, half of it from other companies in the Mitsubishi group. US-German carmaker DaimlerChrylser, a 30% shareholder in Mitsubishi Motors, decided in April 2004 not to pump in any more money. The deal with Peugeot was celebrated by Mitsubishi's newly-appointed chief executive Takashi Nishioka, who took over after three top bosses stood down last month to shoulder responsibility for the firm's troubles. Mitsubishi Motors has forecast a net loss of 472bn yen in its current financial year to March 2005. Last month, it signed a production agreement with Japanese rival Nissan Motor to supply it with 36,000 small cars for sale in Japan. It has been making cars for Nissan since 2003.
    ori_questions = [
    "Which two car manufacturers have signed a Memorandum of Understanding?",
    "How many SUVs will Mitsubishi supply to Peugeot?",
    "What kind of engines will be used in the SUVs?",
    "Where will the SUVs be built?",
    "Which market will the SUVs be sold in primarily?",
    "What has been the impact on Mitsubishi's sales in the past year?",
    "Why has Mitsubishi sought financial bailouts?",
    "How much additional financial backing is Mitsubishi looking for?",
    "What was the decision made by DaimlerChrylser regarding Mitsubishi Motors?",
    "What is Mitsubishi's forecasted net loss for the current financial year?"
    ]
    """confused questions
    1. Which two car manufacturers have signed a partnership agreement?
    2. How many sedans will Peugeot supply to Mitsubishi?
    3. What kind of transmissions will be used in the SUVs?
    4. Where will the SUVs be sold?
    5. Which market will the sedans be sold in primarily?
    6. What has been the impact on Peugeot's sales in the past year?
    7. Why has Peugeot sought financial bailouts?
    8. How much additional financial backing is Peugeot looking for?
    9. What was the decision made by Mitsubishi Motors regarding Peugeot?
    10. What is Peugeot's forecasted net gain for the current financial year?"
    """
    for text in texts:
        print(f"triples of Text: {text}\n")
        triples = extract_triples_compact_ie(text)
        for triple in triples:
            print(f"subject: {triple[0]}, relation: {triple[1]}, object: {triple[2]}")

if __name__ == "__main__":
    main()