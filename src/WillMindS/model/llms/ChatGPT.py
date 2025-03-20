import json
import os

import openai

class ChatGPT():
    def __init__(self,organization,api_key):
        openai.organization = organization
        openai.api_key = api_key
        self.model_name = 'ChatGPT'

    def answer(self,input_text):
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-4", #model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": input_text}
                ]
            )
            output_text = completion.choices[0].message['content'].encode('utf-8').decode('utf-8')
        except:
            completion = None
            output_text = ''
        return completion, output_text

    def chat(self):
        pass