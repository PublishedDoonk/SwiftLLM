from .genai_wrapper import LanguageModel
import openai
import requests
import os
import json

OPENAI_TOKEN_PRICES = {
    'gpt-4o': [0.000005, 0.000015],
    'gpt-4o-2024-05-13': [0.000005, 0.000015],
    'gpt-3.5-turbo-0125': [0.0000005, 0.0000015],
    'gpt-3.5-turbo-instruct': [0.0000015, 0.000002],
    'gpt-4-turbo': [0.00001, 0.00003],
    'gpt-4-turbo-2024-04-09': [0.00001, 0.00003],
    'gpt-4': [0.00003, 0.00006],
    'gpt-4-32k': [0.00006, 0.00012],
    'gpt-4-0125-preview': [0.00001, 0.00003],
    'gpt-4-1106-preview': [0.00001, 0.00003],
    'gpt-4-vision-preview': [0.00001, 0.00003],
    'gpt-3.5-turbo-1106': [0.000001, 0.000002],
    'gpt-3.5-turbo-0613': [0.0000015, 0.000002],
    'gpt-3.5-turbo-16k-0613': [0.000003, 0.000004],
    'gpt-3.5-turbo-0301': [0.0000015, 0.000002],
    'davinci-002': [0.000002, 0.000002],
    'babbage-002': [0.0000004, 0.0000004],
}

class OpenAI(LanguageModel):
    def __init__(self, instructions: str = None, sample_outputs: list = None, schema: dict = None, prev_messages: list = None, response_type: str = None, model: str = 'gpt-3.5-turbo', streaming=False, organization: str = '', project: str = '', api_key: str = None):
        if os.getenv('OPENAI_API_KEY') is None and api_key is None:
            raise KeyError('OPENAI_API_KEY not found in environment variables. Please set the OPENAI_API_KEY environment variable to use the OpenAI models.')
        if api_key is None:
            api_key = os.environ['OPENAI_API_KEY']
        self.project = project
        self.organization = organization
        self.model = model
        self.streaming = streaming
        super().__init__(instructions, sample_outputs, schema, prev_messages, response_type)
        if self.response_type == 'JSON' and self.no_json_capability():
            raise TypeError(f'The model {self.model} does not support JSON output. Please choose a model that supports JSON output. The following base models and their offshoots support JSON response: gpt-4o, gpt-4-turbo, gpt-3.5-turbo.')
        client_kwargs = self.get_client_kwargs(api_key)
        self.client = openai.OpenAI(**client_kwargs)
        self.format_instructions()
    
    def get_client_kwargs(self, api_key):
        """
        This function gets a dict of kwargs to pass to the OpenAI client.
        """
        kwargs = {'api_key': api_key}
        if self.organization:
            kwargs['organization'] = self.organization
        if self.project:
            kwargs['project'] = self.project
        
        return kwargs
    
    def no_json_capability(self):
        """
        Returns false if self.model is capable of generating JSON strings and True if it is not.
        """
        json_capable_models: list[str] = ['gpt-4o', 'gpt-4-turbo', 'gpt-3.5-turbo']
        for model in json_capable_models:
            if model in self.model.lower(): # the model chosen is capable of returning JSON strings
                return False 
        return True # the model chosen is not capable of returning JSON strings
        
    def format_instructions(self):
        """
        This method formats the instructions as the first message in prev_messages.
        """
        self.format_messages(role='system', content=self.instructions)
        self.format_messages(role='assistant', content='OK. I will follow the system instructions to the best of my ability.')
    
    def calculate_inference_cost(self, response: str):
        """
        This method calculates the cost of the inference call and stores it in the activity log.
        """
        self.activity_log[-1]['total_inference_cost'] = OPENAI_TOKEN_PRICES[self.model][0] * response.usage.prompt_tokens
        self.last_inference_cost = OPENAI_TOKEN_PRICES[self.model][1] * response.usage.completion_tokens
    
    def parse_content(self, response: str):
        """
        This method parses the content from the response and stores it in the activity log.
        """
        if isinstance(response, openai.Stream):
            content = self.handle_stream(response)
        else:
            content = response.choices[0].message.content
        
        self.format_messages(role='assistant', content=content)
        
        return content
    
    def process_response(self, response, content):
        """
        This method processes the response based on the response_type and schema.
        """
        if self.response_type == 'RAW':
            return response
        if self.response_type == 'CONTENT':
            return content
        
        response = json.loads(content)
        if self.schema == {} or self.validate_response_schema(response):
            return response
    
    def generate(self, prompt: str, max_tokens: int = 1024):
        """
        This method generates a response from the OpenAI model given a prompt.
        """
        if self.response_type == 'JSON':
            prompt = f'Input: {prompt}\n\nOutput JSON Schema:\n{json.dumps(self.schema)}\n\nList of Sample Outputs:\n{json.dumps(self.sample_outputs)}'
        self.format_messages(role='user', content=prompt)
        
        kwargs = self.get_kwargs(max_tokens)
        response = self.get_response(kwargs)
        self.calculate_inference_cost(response)
        content = self.parse_content(response)
                
        return self.process_response(response, content)
    
    def get_response(self, kwargs: dict):
        """
        Generate a response to the prompt using the OpenAI API. Return the suitable response based on the response_type.
        """
        response = self.client.chat.completions.create(**kwargs)
        if response.model:
            self.model = response.model
        
        return response
    
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
        if self.streaming == True: 
            kwargs['stream'] = True
        
        # take advantage of OpenAI's JSON output mode
        if self.response_type == 'JSON': 
            kwargs['response_format'] = {'type': 'json_object'}
        
        return kwargs
    
    def handle_stream(self, response):
        """
        This method handles the stream response from the OpenAI API.
        """
        content = ''
        for chunk in response:
            if (new_tokens := chunk.choices[0].delta.content) is not None:
                content = content + new_tokens
                print(new_tokens, end='')
        
        return content
    
    