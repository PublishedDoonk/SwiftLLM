from .genai_wrapper import LanguageModel
import openai
import requests
import os
import json

class OpenAI(LanguageModel):
    def __init__(self, instructions: str = None, sample_outputs: list = None, schema: dict = None, prev_messages: list = None, response_type: str = None, model: str = 'gpt-3.5-turbo', streaming=False, organization: str = '', project: str = '', api_key: str = None):
        if os.getenv('OPENAI_API_KEY') is None and api_key is None:
            raise KeyError('OPENAI_API_KEY not found in environment variables. Please set the OPENAI_API_KEY environment variable to use the OpenAI models.')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
        self.project = project
        self.organization = organization
        self.model = model
        self.streaming = streaming
        super().__init__(instructions, sample_outputs, schema, prev_messages, response_type)
        self.client = openai.OpenAI()
        self.format_instructions()
    
    def format_messages(self, role: str, content: str):
        self.prev_messages.append({'role': role, 'content': content})
        
    def format_instructions(self):
        """
        This method formats the instructions as the first message in prev_messages.
        """
        self.prev_messages.append({'role': 'system', 'content': self.instructions})
        self.prev_messages.append({'role': 'assistant', 'content': 'OK. I will follow the system instructions to the best of my ability.'})
    
    def generate(self, prompt: str, max_tokens: int = 1024):
        """
        This method generates a response from the OpenAI model given a prompt.
        """
        if self.response_type == 'JSON':
            prompt = f'Input: {prompt}\n\nOutput JSON Schema:\n{json.dumps(self.schema)}\n\nList of Sample Outputs:\n{json.dumps(self.sample_outputs)}'
        self.format_messages(role='user', content=prompt)
        
        kwargs = self.get_kwargs(max_tokens)
        
        return self.get_response(kwargs)
    
    def get_kwargs(self, max_tokens: int):
        """
        Build out all the arguments needed for the OpenAI API call based on the model's properties.
        """
        # initialize the kwargs dictionary with model, messages, and max_tokens
        kwargs = {
            'model': self.model,
            'messages': self.prev_messages,
            'max_tokens': max_tokens,
        }
        
        # if streaming is truthy, set stream to true
        if self.streaming == 1: 
            kwargs['stream'] = True
        
        # take advantage of OpenAI's JSON output mode
        if self.response_type == 'JSON': 
            kwargs['response_format'] = {'type': 'json_object'}
        
        return kwargs
        
        
    def get_response(self, kwargs: dict):
        """
        Generate a response to the prompt using the OpenAI API. Return the suitable response based on the response_type.
        """
        response = self.client.chat.completions.create(**kwargs)
        
        self.format_messages(role='assistant', content=response.choices[0].message.content)
        
        if self.response_type == 'RAW':
            return response
        if self.response_type == 'CONTENT':
            return response.choices[0].message.content
        
        # read response as JSON and return results if it matches the schema or there is no schema
        response = json.loads(response.choices[0].message.content)
        if self.schema == {} or response.keys() == self.schema.keys():
            return response
        
        raise KeyError(f'OpenAI generated response does not match the schema. Expected keys: {self.schema.keys()}. Response keys: {response.keys()}. If problem persists, try setting a simpler schema or revising system instructions.')
        
        
        