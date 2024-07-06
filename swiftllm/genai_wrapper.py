import json
from datetime import datetime
from requests import Response

class LanguageModel:
    
    """
    This is the base class for all the generative AI model wrappers in this package. It implements the
    logic for storing system instructions, previous prompts and responses, and the desired response
    schema.
    """
    
    def __init__(self, instructions: str, sample_outputs: list | None = None, schema: dict | None = None, prev_messages: list | None = None, response_type: str = None):
        """
        Initialize the LanguageModel object.

        Args:
            instructions (str): System instructions for the generative AI model.
            sample_outputs (list | None, optional): Examples of good outputs the model should strive to generate. Defaults to None.
            schema (dict | None, optional): Schema that defines object the model should generate in JSON mode. Defaults to None.
            prev_messages (list | None, optional): List of previous messages. Defaults to None.
            response_type (str, optional): Format the model should return as a response. Valid options: ('JSON', 'CONTENT', 'RAW'). Defaults to None.
        """
        self.activity_log: list = [] # store all prompts, responses, and exceptions in the order they occur
        self.total_inference_cost: float = 0.0 # store the total cost of all inference calls
        
        if not instructions or not isinstance(instructions, str):
            instructions = 'You will be provided instructions/prompts. Do your best to generate an appropriate response to the prompts.'
        self.instructions = instructions # store system instructions
        
        # initialize or set sample outputs
        if not isinstance(sample_outputs, list):
            sample_outputs = []
        self.sample_outputs = sample_outputs
        
        # intialize or set output schema
        if not isinstance(schema, dict):
            schema = {}
        self.schema = schema
        
        # initialize or set previous messages
        if not isinstance(prev_messages, list):
            prev_messages = []
        self.prev_messages = prev_messages
        
        # predict or set response type
        if self.response_type_invalid(response_type):
            response_type = self.predict_response_type()
        self.response_type = response_type.upper()
        
        self.format_instructions()
        
    def predict_response_type(self):
        """
        This function predicts what the response_type should be if one isn't provided.
        Response type is JSON if a schema is provided, otherwise it is CONTENT.
        """
        if self.schema:
            return 'JSON'
        return 'CONTENT'
        
    def response_type_invalid(self, response_type: str):
        """
        Returns true if the response_type is not a string in ['JSON', 'CONTENT', 'RAW'], case insensitive.
        """
        if not isinstance(response_type, str): # non-string response type is invalid
            return True
        return response_type.upper() not in ['JSON', 'CONTENT', 'RAW'] # return true if response type is not a valid option
    
    def format_instructions(self):
        """
        Creates a final system instructions message that includes the system instructions, schema, and sample outputs.
        """
        self.instructions = 'SYSTEM INSTRUCTIONS:\n' + self.instructions.strip()
        if self.schema:
            self.instructions = self.instructions + '\n\nReturn your output as a correct JSON string. It should be loadable with the python command json.loads(output).\n\nOutput Schema:\n' + json.dumps(self.schema)
        if self.sample_outputs:
            self. instructions = self. instructions + '\n\n'.join([f'Sample Output {i+1}:\n{json.dumps(s)}' for i, s in enumerate(self.sample_outputs)])
            
    def prompt(self, prompt: str, retries: int = 3, max_tokens: int = 1024):
        """
        This method calls the generate method and handles any exceptions that occur. It will retry the generate method up to the number of retries specified
        if an exception occurs. 
        """
        for _ in range(retries):
            try:
                self.log_activity(prompt, role='user')
                response = self.generate(prompt, max_tokens)
                self.log_activity(response, role='assistant')
                return response
            except Exception as e:
                self.log_activity(e, role='system')
        
        return None

    def log_activity(self, message: any, role: str):
        """
        This method logs the message to the activity log. It includes the message, the timestamp, and the total inference cost.
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if isinstance(message, (dict, list)):
            message = json.dumps(message)
        if isinstance(message, Response):
            try:
                message = message.text
            except:
                message = 'model generated Response object with status code: ' + message.status_code
        if not isinstance(message, str):
            message = str(message)
        self.activity_log.append({'timestamp': timestamp, 'role': role, 'message': message, 'total_inference_cost': self.total_inference_cost})
    
    def parse_json(self, text: str):
        """This function finds a JSON string within the provided text. If there is no JSON found in the text, it raises an Exception.

        Args:
            text (str): the body of text to look for JSON substrings in.
        """
        pass
    
    def generate(self, prompt, max_tokens):
        """
        This method should be implemented in the child class of the LanguageModel class. It should handle the logic to actually pass the prompt to the model and return the 
        response to the prompt method. The response should match the response_type specified in the constructor.
        
        Args:
            prompt (str): The prompt to pass to the model.
            max_tokens (int): The maximum number of tokens the model should generate.
        """
        error_message = """
        The generate method must be implemented in child class. This method should handle the logic to actually pass the prompt to the model and return the response to the prompt method. 
        Response should match the response_type specified in the constructor.
        """
        raise NotImplementedError(error_message)
    
    def display_activity_log(self):
        """
        This method pretty prints the activity log to the console for troubleshooting purposes.
        """
        print('ACTIVITY LOG:')
        print('-------------')
        print(f'{"Timestamp":<25}{"Role":<15}{"Cost":<6}{"Message":<60}')
        for entry in self.activity_log:
            print(f"{entry['timestamp']:<25}{entry['role']:<15}{entry['total_inference_cost']:<6}{entry['message']:<60}")
    
    def validate_response_schema(self, response: dict, schema: dict = None):
        """This method validates the response against the schema provided in the constructor. If the response is not valid, it raises an exception.

        Args:
            response (dict): AI generated JSON response
            
        Raises:
            KeyError: If the response does not match the schema provided in the constructor.
        """
        if schema is None:
            schema = self.schema
            
        if schema.keys() != response.keys():
            raise KeyError(f'Generated response does not match the schema. Expected keys: {schema.keys()}. Response keys: {response.keys()}. If problem persists, try setting a simpler schema or revising system instructions.')
        
        for k in schema.keys():
            if isinstance(schema[k], dict):
                self.validate_response_schema(response=response[k], schema=schema[k])
            if isinstance(schema[k], list) and not isinstance(response[k], list):
                raise ValueError(f'Expected a list for key {k} in response. Received {type(response[k])} instead.')
            if not schema[k] or not response[k] or not isinstance(schema[k], list):
                continue
            if isinstance(schema[k][0], dict):
                for obj in response[k]:
                    self.validate_response_schema(response=obj, schema=schema[k][0])