# SwiftLLM

## Overview

&emsp;&emsp;Large Language Models (LLM) are one of the most remarkable advancements in AI in the last few years. These models are capable of generating all kinds of useful content based on not only text, but now with all sorts of multimodal input. There are a plethora of LLM models available from several different providers: X, Meta, OpenAI, and groq to name a few. Each provider has their own API with their own style of doing things. There isn't a uniform way to access these different providers which makes accessing the various pros and cons associated with each provider fairly cumbersome.

&emsp;&emsp;There are several problems I've encountered when working with these models to solve unstructured text data related problems for customers. First, each API for the major model providers has its own request formatting and you need to understand how to manipulate HTTP requests and JSON objects to access the models. Second, most of these APIs don't have a good way to track exactly how much your inference cost is for the requests you've used. Their websites have pricing info, but the API only tells you how many tokens you've used, then you have to do that math on your own. Weaksauce. Third, many LLMs are plenty capable of generating strings in JSON format, but their API doesn't offer this explicitly as a feature (aside from OpenAI that is). Finally, these APIs don't offer an easy way to track your prompts, responses, and errors that may have come up while running. That means you have to implement that yourself!

&emsp;&emsp;This library aims to tackle the problems listed above (and any other problems I can fix). First, I want to abstract away the HTTP requests associated with accessing models from different providers and develop a provider agnostic framework that simplifies accessing all the major APIs. Second, this library will implement a way to track inference costs per message and in aggregate. Third, this library will provide a "response_type" argument that can be set to "RAW", "JSON", or "CONTENT", so that the model can generate valid python objects, a string containing the generation, or for power users the default response object from the API. Lastly, there will be an activity log that tracks all prompts, responses, and errors during model prompting. 

## Getting Started

This project is still in its infancy, but it is available on PyPI.


To install the library install it with pip.

<code>pip install SwiftLLM</code>

Before you are able to make use of the package, you will still need an API key for whichever model in the library you are interested in using. Currently, only OpenAI generative AI models are supported, but other model provider APIs will be added as time allows.

The API key needs to either be saved in the runtime environment with the default name for that API or passed in as a "key" argument to the model during construction.

For OpenAI the api key needs to be saved as an env variable called OPENAI_API_KEY.

```.env
OPENAI_API_KEY="<your key here>"
```

Once all the API keys you need are configured, you can begin using the LanguageModel objects available in the library. Each model is a child of the LanguageModel class which implements a lot of the core functions that may be needed regardless of which model is chosen, such as JSON validation, the prompt method, saving messages and errors, etc.. You can extend this LanguageModel to develop your own LLM wrapper or import one of the wrappers available in the library (currently only OpenAI).

This is how simple it is to get a reply from ChatGPT using this library.

```python
from swiftllm import OpenAI

model = OpenAI(instructions="have a chat with me")
reply = model.prompt("Hello, how's your day going?")
print(reply)
```

In the sample above, the model object is an OpenAI object with the system instructions "have a chat with me", the response_type is "CONTENT" by default meaning its response will just be a string, the default model is gpt-3.5-turbo, and it handles all the HTTP request and response parsing behind the scenes. 

OpenAI Note: If you have multiple projects and organizations, you will have to pass them in as arguments. Nothing I can do about that.

A more complicated and useful use case would be using an LLM to generate structured output from an unstructured body of text. The LanguageModel object has two optional arguments: schema and sample_outputs. Schema is a python dictionary that describes the type of object you want the LLM to return to you. The sample outputs is a list of examples containing "good" outputs from the model.

```python
from swiftllm import OpenAI

schema: dict = {
    'people': [
        {
            'name': 'string',
            'age': 'int',
            'title': 'string',
        }
    ]
}

sample_outputs: list = [
    {
        'people': [
            {
                'name': 'John Doe',
                'age': 27,
                'title': 'HR Manager',
            },
            {
                'name': 'Sally Sue',
                'age': 18,
                'title': 'Accountant',
            }
        ]
    },
    {
        'people': [
            {
                'name': 'Cathy Stan',
                'age': 59,
                'title': 'Chief Executive Officer',
            }
        ]
    }
]

instructions: str = 'Find all the people in the given body of text, their age, and what their job title is.'

target_text: str = 'Sheryll Marsh, a 49 year old lawyer from Ohio, was traveling through the hills of Kentucky when something changed her life forever. She found and married her husband Jimmy Dean, a 50 year old mechanic from Delaware.'

model = OpenAI(instructions=instructions, schema=schema, sample_outputs=sample_outputs) # if a schema or sample outputs is provided, JSON response type is automatically selected
people = model.prompt(target_text)

for person in people['people']:
    print('name', person['name'])
    print('age', person['age'])
    print('title', person['title'])

```

In the above example, the LanguageModel object will prompt the model API (in this case gpt-3.5-turbo), and have it generate a JSON string that matches the schema provided. The output will return the name, age, and title for the two people in the target text. All the JSON string parsing and converting to a valid python object is done for you by the model. If the JSON object generated by the model is not valid, it will retry generation until it succeeds or retries are exceeded. Retries is 3 by default, but can be passed in as an argument to the model or as an optional argument to the prompt method. 

OpenAI Note: OpenAI provides JSON mode as a feature in their API for all models that support it. Therefore, I've designed the OpenAI object to use that feature when it is available and ignore JSON as a response_type when it is not. If a model that doesn't support JSON is selected, and JSON is specified as the response type, the response_type will default to "CONTENT".