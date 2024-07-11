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
    
    def __init__(self, instructions: str = None, sample_outputs: list = None, schema: dict = None, prev_messages: list = None, response_type: str = None, model: str = 'mixtral-8x7b', api_key: str = None, temperature: float = 0.5, max_tokens: int = 1024, top_p: float = 1.0, stop: str = None, stream: bool = False):
        super().__init__(instructions, sample_outputs, schema, prev_messages, response_type)
        if api_key is None:
            api_key = os.environ.get('GROQ_API_KEY')
        self.model = find_model(model)
        self.client = groq.Groq(api_key=api_key)
        self.format_instructions()
        self.initialize_messages()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.stop = stop
        self.stream = stream
    
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
        
    def get_completion_kwargs(self, max_tokens: int, temperature: float, top_p: float, stop: str, stream: bool):
        """This function gets all the kwargs for the chat.completion.create method in the Groq API client.

        Args:
            max_tokens (int): max number of tokens generated by the model
            temperature (float): randomness of model, 0 = deterministic, 1 = very random
            top_p (float): controls diversity via nucleus sampling
            stop (str): stop sequence so model knows to stop generating.
            stream (bool): return stream of chat completion deltas as response generates or not

        Returns:
            dict: kwargs for the chat.completion.create method in the Groq API client
        """
        args = [max_tokens, temperature, top_p, stop, stream]
        attributes = [self.max_tokens, self.temperature, self.top_p, self.stop, self.stream]
        for arg, attribute in zip(args, attributes):
            if arg is None:
                arg = attribute
                
        keys = ['max_tokens', 'temperature', 'top_p', 'stop', 'stream']
        kwargs = {key: arg for key, arg in zip(keys, args) if arg}
        
        kwargs['model'] = self.model
        kwargs['messages'] = self.prev_messages
        
        return kwargs
        
    def generate(self, prompt: str, response_type: str = None, max_tokens: int = None, temperature: float = None, top_p: float = None, stop: str = None, stream: bool = None):
        """This prompts the model to generates a response from the Groq API.
        
        Args:
            prompt: str - the prompt to generate a response from
            response_type: str [opt] - the type of response to return (RAW, CONTENT, JSON)
            max_tokens: int [opt] - the maximum number of tokens the model should generate
            temperature: float [opt] - the randomness of the model, 0 = deterministic, 1 = very random
            top_p: float [opt] - controls diversity via nucleus sampling
            stop: str [opt] - stop sequence so model knows to stop generating
            stream: bool [opt] - return stream of chat completion deltas as response generates or not
            
        Returns:
            str | response | dict: the response from the Groq API (type depends on response_type provided)
        """
        self.format_messages(role='user', content=prompt) # adds prompt as user message
        kwargs: dict = self.get_completion_kwargs(max_tokens, temperature, top_p, stop, stream) # get kwargs for chat completion create method
        response = self.client.chat.completions.create(**kwargs) # generate response from Groq API
        content = self.process_response(response, response_type) # return appropriate value based on response_type
        
        return content
    
    def process_response(self, response, response_type: str):
        """
        Process the generated response from the Groq API and return the appropriate value corresponding with the response_type
        """
        print(response_type)
        if response_type is None:
            response_type = self.response_type
        print(response_type)
        #print(response_type)
            
        if response_type == 'RAW':
            return response
        
        content = response.choices[0].message.content
        if response_type == 'CONTENT':
            return content
        
        json_obj = self.parse_json_content(content)
        return json_obj
    