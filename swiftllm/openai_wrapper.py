from .genai_wrapper import LanguageModel
import openai
import requests
import os

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
        self.format_messages(role='user', content=prompt)
        kwargs = {
            'model': self.model,
            'messages': self.prev_messages,
            'max_tokens': max_tokens,
        }
        if self.streaming == 1: # if streaming is truthy, set stream to true
            kwargs['stream'] = True
        
        if self.response_type == 'JSON': # take advantage of OpenAI's JSON output mode
            kwargs['type'] = 'json_object'
        
        response = self.client.chat.completions.create(**kwargs)
        self.format_messages(role='assistant', content=response.choices[0].message.content)
        if self.response_type == 'RAW':
            return response
        if self.response_type == 'CONTENT':
            return response.choices[0].message.content
        
        
    
        
