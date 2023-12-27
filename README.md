# Studio LLM Service

Built an end-user configurable Generative AI Studio, which can handle multiple input types, use various large language models (LLMs) locally or via an API, supports multiple speech-to-text models, etc.

It has three main components built with Python, the UI, the Middleware service and the LLM Service.

This repository has the code for the LLM service, which is responsible for handling everything from loading models from huggingface to fetching and preparing context from Vector Databases.
