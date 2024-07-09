from .genai_wrapper import LanguageModel
from groq import Groq
import requests
import os
import json

class Groq(LanguageModel):
    
    def __init__(self, instructions: str = None, sample_outputs: list = None, schema: dict = None, prev_messages: list = None, response_type: str = None, model: str = 'mixtral-8x7b', api_key: str = None):
        super().__init__(instructions, sample_outputs, schema, prev_messages, response_type)
        if api_key is None:
            api_key = os.environ.get('GROQ_API_KEY')
        self.model = model
        self.client = Groq(api_key=api_key)
        self.format_instructions()
        self.initialize_messages()
        
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
    
    def process_response(response):
        """
        Process the generated response from the Groq API and return the appropriate value corresponding with the response_type
        """
        
        if self.response_type == 'RAW':
            return response
        if self.response_type == 'CONTENT':
            return response.choices[0].message.content
        return # TO-DO: parse JSON from response, validate schema, and return dict if valid
    