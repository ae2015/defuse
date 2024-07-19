from openai import OpenAI
import os, time, requests, json, re
from collections import defaultdict

class LLMException(Exception):
    pass

class LLM:
    registry = defaultdict(lambda: None)
    @classmethod
    def get(cls, name):
        """
        Get an LLM instance by name from the registry
        """
        return cls.registry[name]

    def __init__(self, name, model, url, headers, parameters):
        """
        Creates a new LLM instance that connects to actual remote LLM using REST API

        Parameters
        ----------
        name : str
            Short name, to use in CSV tables and as key in the registry
        model : str
            Full official name of the LLM model, used in REST calls
        url : str
            Web address of the REST server
        headers : dict
            Provided at REST API calls, in particular contains the secret API key
        parameters : dict
            LLM inference (hyper-)parameters, provided at calls

        Returns
        -------
        A new LLM instance (also available through the registry)
        """

        LLM.registry[name] = self
        self.name = name
        self.model = model
        self.url = url
        self.headers = headers
        self.parameters = parameters

    def __repr__(self):
        return (
            f"LLM(name = '{self.name}', model = '{self.model}', " +
            f"url = {self.url}, headers = {self.headers}, " +
            f"parameters = {self.parameters})"
        )
    
    def __str__(self):
        return self.name

    def __call__(self, prompt):
        """
        Issues a POST call to perform LLM inference

        Parameters
        ----------
        prompt : str | list | dict
            - If `str`, this is a single-turn user's prompt, expecting AI assistant's reply
            - If `dict`, this is a single-turn and should be given to the LLM as-is
            - If `list` of `str`, this is a multi-turn conversation between user and assistant,
              the last turn must be user's
            - If `list` of `dict`, this is a multi-turn and should be given to the LLM as-is

        Returns
        -------
            str
        """
        messages = None
        if isinstance(prompt, str):
            messages = [
                {
                    "role" : "user",
                    "content" : prompt,
                },
            ]
        elif isinstance(prompt, dict):
            messages = [prompt]
        elif isinstance(prompt, list):
            messages = [
                entry if isinstance(entry, dict) else {
                    "role" : "user" if (len(prompt) - i) % 2 == 1 else "assistant",
                    "content" : entry
                } for i, entry in enumerate(prompt)
            ]
        else:
            raise LLMException(f"Incompatible prompt: {prompt}")

        # print(f"\nMessages:\n{messages}\n\n")

        json_data = {
            "model" : self.model,
            "messages" : messages
        }
        json_data.update(self.parameters)
        start_time = time.time()
        response = requests.post(self.url, headers = self.headers, json = json_data)
        end_time = time.time()
        duration = end_time - start_time
        if response.status_code != 200:
            raise LLMException(
                f"Model = {self.model}, Status Code = {response.status_code}, Duration = {duration}\n" +
                f"Prompt: {prompt}\nResponse: {response.json()}" # {json.dumps(response)}"
            )
        # print("text_output = " + str(response.json()))
        text_output = response.json()["choices"][0]["message"]["content"]
        # print(f"Response from OpenAI: {response.json()}\n")
        return text_output



openai_url = "https://api.openai.com/v1/chat/completions"
runpod_llama3_url = f"https://api.runpod.ai/v2/{os.environ.get("RUNPOD_ENDPOINT_ID")}/openai/v1/chat/completions"
openai_headers_1 = {
    "Content-Type": "application/json",
    "Authorization": "Bearer " + os.getenv("OPENAI_API_KEY", ""),
    "OpenAI-Organization": os.getenv("OPENAI_ORGANIZATION_ID", ""),
    "OpenAI-Project": os.getenv("OPENAI_PROJECT_ID", "")
}
openai_headers_2 = {
    "Content-Type": "application/json",
    "Authorization": "Bearer " + os.getenv("OPENAI_API_KEY", "")
}
runpod_llama3_headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer " + os.getenv("RUNPOD_API_KEY", "")
}
gpt_3_dot_5 = LLM(
    name = "gpt-3.5",
    model = "gpt-3.5-turbo",
    url = openai_url,
    headers = openai_headers_1,
    parameters = {
        "temperature": 0.7
    }
)
gpt_4o = LLM(
    name = "gpt-4o",
    model = "gpt-4o",
    url = openai_url,
    headers = openai_headers_2,
    parameters = {
        "temperature": 0.7
    }
)

llama3_8B_in = LLM(
    name = "llama3-8B-in",
    model = "meta-llama/Meta-Llama-3-8B-Instruct",
    url = runpod_llama3_url,
    headers = runpod_llama3_headers,
    parameters = {
        "temperature": 0.7
    }
)
