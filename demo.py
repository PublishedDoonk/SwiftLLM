from swiftllm import Groq
from dotenv import load_dotenv

load_dotenv() # load my Groq API key from the .env file

# Create a SwiftLLM LanguageModel object that communicates with the Groq API
model = Groq(
    instructions='Find all the names, ages, and titles in the text provided.',
    schema={'name':'str', 'age': 'int', 'title': 'str'},
    model='mixtral'
)

target_text = 'Zachary Ivie is a 29 year old data scientist currently looking for new opportunities.'
response = model.prompt(target_text)
print(response)
