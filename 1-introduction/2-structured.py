import os
import typing

from google import genai
from google.genai import types
from pydantic import BaseModel

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Step 1: Define the response format in a Pydantic model


class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


# Step 2: Call the model

completion = client.models.generate_content(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(
        system_instruction="Extract the event information.",
        response_mime_type="application/json",
        response_schema=CalendarEvent,
    ),
    contents="Alice and Bob are going to a science fair on Friday",
)

print(completion.text)

event: CalendarEvent = typing.cast(CalendarEvent, completion.parsed)
event.name
event.date
event.participants
