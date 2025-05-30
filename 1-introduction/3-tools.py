import json
import os

import requests
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# Define the tool (function) we want to call


def get_weather(latitude, longitude):
    """This is a publically available API that returns the weather for a given location."""
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    data = response.json()
    return data["current"]


# Step 1: Call model with get_weather tool defined


get_weather_declaration = {
    "name": "get_weather",
    "description": "Gets the current temperature for provided coordinates.",
    "parameters": {
        "type": "object",
        "properties": {
            "latitude": {
                "type": "number",
                "description": "The latitude coordinate of the location",
            },
            "longitude": {
                "type": "number",
                "description": "The longitude coordinate of the location",
            },
        },
        "required": ["latitude", "longitude"],
    },
}

system_instruction = "You are a helpful weather assistant.  When users ask about weather in cities, use the get_weather function with approximate coordinates for major cities. "

contents = [
    types.Content(
        role="user", parts=[types.Part(text="What's the weather like in Paris today?")]
    )
]

model = "gemini-2.0-flash"

tools = types.Tool(function_declarations=[get_weather_declaration])  # type: ignore

config = types.GenerateContentConfig(
    tools=[tools], system_instruction=system_instruction
)

completion = client.models.generate_content(
    model=model, config=config, contents=contents
)

completion.model_dump


# Step 2: Execute get_wether function


if completion.candidates[0].content.parts[0].function_call:  # type: ignore
    function_call = completion.candidates[0].content.parts[0].function_call  # type: ignore
    print(f"Function to call: {function_call.name}")
    print(f"Arguments: {function_call.args}")

else:
    print("No function call found in the response.")
    print(completion.text)


def call_function(name, args):
    if name == "get_weather":
        return get_weather(**args)


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


# Step 3: Supply result and call model again


class WeatherResponse(BaseModel):
    temperature: float = Field(
        description="The current temperature in celsius for the given location."
    )
    response: str = Field(
        description="A natural language response to the user's question."
    )


config2 = types.GenerateContentConfig(
    system_instruction=system_instruction,
    response_schema=WeatherResponse,
    response_mime_type="application/json",
)

completion2 = client.models.generate_content(
    model=model, config=config2, contents=contents
)

final_response = completion2.parsed
print(final_response.response)  # type: ignore
print(final_response.temperature)  # type: ignore
