from .genai_wrapper import LanguageModel
import groq
import requests
import traceback
import sys
import os
import json

SUPPORTED_GROQ_MODELS: list[str] = [
    'mixtral-8x7b-32768',
    'llama3-70b-8192',
    'llama3-8b-8192',
    'gemma2-9b-it',
    'gemma-7b-it',
    
]

def find_model(model: str):
    """
    This function finds a supported model that matches the input model and raises an error if there is no matching model.
    """
    for supported_model in SUPPORTED_GROQ_MODELS:
        if match_model(model, supported_model):
            return supported_model
        
    raise ValueError(f'The model {model} is not supported by the Groq API. Please choose a supported model from the following list: {SUPPORTED_GROQ_MODELS}')
        
def match_model(model: str, supported_model: str):
    """
    This function finds a supported model that matches the input model and raises an error if there is no matching model.
    """
    model = model.lower().replace('-', ' ').split(' ')
    for word in model:
        if word not in supported_model.lower():
            return False
    return True
    
class Groq(LanguageModel):
    
    def __init__(self, instructions: str = None, sample_outputs: list = None, schema: dict = None, prev_messages: list = None, response_type: str = None, model: str = 'mixtral-8x7b', api_key: str = None):
        super().__init__(instructions, sample_outputs, schema, prev_messages, response_type)
        if api_key is None:
            api_key = os.environ.get('GROQ_API_KEY')
        self.model = find_model(model)
        self.client = groq.Groq(api_key=api_key)
        self.format_instructions()
        self.initialize_messages()
    
    def find_model(self, model: str):
        """
        This method finds a supported model that matches the input model and raises an error if there is no matching model.
        """
        for supported_model in SUPPORTED_GROQ_MODELS:
            if model.lower() in supported_model.lower():
                return supported_model
    
    def initialize_messages(self):
        """
        This method initializes the messages in the prev_messages list.
        """
        self.format_messages(role='system', content=self.instructions)
        self.format_messages(role='assistant', content='OK. I will follow the system instructions to the best of my ability.')
        
    def generate(self, prompt: str, max_tokens: int = 1024):
        """
        This method generates a response from the Groq model given a prompt.
        """
        self.format_messages(role='user', content=prompt)
        
        response = self.client.chat.completions.create(
            messages=self.prev_messages,
            model=self.model,
            max_tokens=max_tokens
        )
        
        content = self.process_response(response)
        
        return content
    
    def process_response(self, response):
        """
        Process the generated response from the Groq API and return the appropriate value corresponding with the response_type
        """
        
        if self.response_type == 'RAW':
            return response
        if self.response_type == 'CONTENT':
            return response.choices[0].message.content
        return # TO-DO: parse JSON from response, validate schema, and return dict if valid
    