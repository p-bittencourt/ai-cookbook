import json
import os

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

print(f"Current working directory: {os.getcwd()}")  # Add this line to debug


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# Define the knowledge base retrieval tool


def search_kb(question: str):
    """
    Load the knowledge base from the JSON file.
    (This is a mock function for demonstration purposes, we don't search)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    kb_path = os.path.join(script_dir, "kb.json")
    with open(kb_path, "r") as f:
        return json.load(f)


# Step 1: Call model with search_kb tool defined

tool_functions = [
    {
        "name": "search_kb",
        "description": "Get the answer to the user's question from the knowledge base.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The user's question to search for in the knowledge base",
                },
            },
            "required": ["question"],
        },
    }
]


system_instruction = "You are a helpful assistant that answers questions from the knowledge base about our e-commerce store."

contents = [
    types.Content(role="user", parts=[types.Part(text="What is the return policy?")])
]

tools = types.Tool(function_declarations=tool_functions)  # type: ignore

model = "gemini-2.0-flash"

config = types.GenerateContentConfig(
    tools=[tools], system_instruction=system_instruction
)

completion = client.models.generate_content(
    model=model, config=config, contents=contents
)

print(completion.model_dump)

# Step 2: Execute the search_kb function


def call_function(name, args):
    if name == "search_kb":
        return search_kb(**args)


for part in completion.candidates[0].content.parts:  # type: ignore
    if part.function_call:
        tool_call = part.function_call
        function_result = call_function(tool_call.name, tool_call.args)
        function_response_part = types.Part.from_function_response(
            name=tool_call.name, response={"result": function_result}  # type: ignore
        )

        print(f"Function execution result: {function_result}")

        contents.append(
            types.Content(role="model", parts=[types.Part(function_call=tool_call)])
        )
        contents.append(types.Content(role="user", parts=[function_response_part]))


class KBResponse(BaseModel):
    answer: str = Field(description="The answer to the user's quesiton.")
    source: int = Field(description="The record id of the answer")


config2 = types.GenerateContentConfig(
    system_instruction=system_instruction,
    response_schema=KBResponse,
    response_mime_type="application/json",
)

completion2 = client.models.generate_content(
    model=model, config=config2, contents=contents
)

final_response = completion2.parsed
print(final_response)
