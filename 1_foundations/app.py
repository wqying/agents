from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr


load_dotenv(override=True)

# def push(text):
#     requests.post(
#         "https://api.pushover.net/1/messages.json",
#         data={
#             "token": os.getenv("PUSHOVER_TOKEN"),
#             "user": os.getenv("PUSHOVER_USER"),
#             "message": text,
#         }
#     )


# def record_user_details(email, name="Name not provided", notes="not provided"):
#     push(f"Recording {name} with email {email} and notes {notes}")
#     return {"recorded": "ok"}

# def record_unknown_question(question):
#     push(f"Recording {question}")
#     return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

# tools = [{"type": "function", "function": record_user_details_json},
#         {"type": "function", "function": record_unknown_question_json}]


class Me:

    def __init__(self):
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.name = "Qianying Wong"
        reader = PdfReader("me/Profile.pdf")
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text
        reader2 = PdfReader("me/QianYingWong_CV.pdf")
        reader3 = PdfReader("me/Stanford_PersonalHistory.pdf")
        reader4 = PdfReader("me/USC_PersonalStatement.pdf")
        readers = [reader2, reader3, reader4]
        self.other_info = ""
        for page in readers:
            for p in page.pages:
                text = p.extract_text()
                if text:
                    self.other_info += text
        with open("me/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()


    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results
    
    def system_prompt(self):
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
particularly questions related to {self.name}'s career, background, skills and experience. \
Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer questions. \
You are also given other info like {self.name}'s CV, personal history, and statement of purpose for a grad school application.  \
Note that the grad school application is just an example. Please do not mention which school or program, but feel free to use the contents for context. \
{self.name} is applying to master programs in areas related to AI and computer science, and is interested in working in the industry after graduation. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer to any question, always say 'I don't know how to answer that yet! Please contact the human me directly!'"

        system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.linkedin}\n\n ## Other Information:\n{self.other_info}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}. Do not mention that you are an assisant or AI model."
        return system_prompt
    
    def chat(self, message, history):
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False
        while not done:
            response = self.openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
            if response.choices[0].finish_reason=="tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        return response.choices[0].message.content
    

if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(me.chat, type="messages").launch(share=True)
    