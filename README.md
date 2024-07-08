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