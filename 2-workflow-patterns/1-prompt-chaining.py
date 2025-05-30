import os
import logging

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
model = "gemini-2.0-flash"


# ----------------------------------------------------------
# Step 1 Define the data models for each stage
# ----------------------------------------------------------


class EventExtraction(BaseModel):
    """First LLM call: Extract basic event information"""

    description: str = Field(description="Raw description of the event")
    is_calendar_event: bool = Field(
        description="Whether this text describes a calendar event"
    )
    confidence_score: float = Field(description="Confidence score between 0 and 1")


class EventDetails(BaseModel):
    """Second LLM call: Parse specific event details"""

    name: str = Field(description="Name of the event")
    date: str = Field(
        description="Date and time of the event. Use ISO 8601 to format this value."
    )
    duration_minutes: int = Field(description="Expected duration in minutes")
    participants: list[str] = Field(description="List of participants")


class EventConfirmation(BaseModel):
    """Third LLM call: Generate confirmation message"""

    confirmation_message: str = Field(
        description="Natural language confirmation message"
    )
    calendar_link: Optional[str] = Field(
        description="Generated calendar link if applicable"
    )


# ----------------------------------------------------------
# Step 2: Define the functions
# ----------------------------------------------------------


def extract_event_info(user_input: str) -> EventExtraction:
    """First LLM call to determine if input is a calendar event"""
    logger.info("Starting event extraction analysis")
    logger.debug("Input text %s", user_input)

    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

    contents = types.Content(role="user", parts=[types.Part(text=user_input)])

    config = types.GenerateContentConfig(
        system_instruction=f"{date_context} Analyze if the text describes a calendar event.",
        response_mime_type="application/json",
        response_schema=EventExtraction,
    )

    completion = client.models.generate_content(
        model=model, config=config, contents=contents
    )

    result = completion.parsed

    assert isinstance(
        result, EventExtraction
    ), f"Expected EventExtraction, got {type(result)}"

    logger.info(
        f"Extraction complete - Is calendar event: {result.is_calendar_event}, Confidence: {result.confidence_score}, Description: {result.description}"
    )

    return result


# Function call to test step 1
# extract_event_info("Tomorrow I'm meeting Mark at the central plaza at 3pm.")


def parse_event_details(description: str) -> EventDetails:
    """Second LLM call to extract specific event details"""
    logger.info("Starting event details parsing")

    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

    contents = types.Content(role="user", parts=[types.Part(text=description)])

    config = types.GenerateContentConfig(
        system_instruction=f"{date_context} Extract detailed event information. When dates reference 'next Tuesday' or similar relative dates, us this current date as reference",
        response_mime_type="application/json",
        response_schema=EventDetails,
    )

    completion = client.models.generate_content(
        model=model, config=config, contents=contents
    )

    result = completion.parsed

    assert isinstance(
        result, EventDetails
    ), f"Expected EventDetails, fot {type(result)}"

    logger.info(
        f"Parsed event details - Name: {result.name}, Date: {result.date}, Duration: {result.duration_minutes}m"
    )

    logger.debug(f"Participants: {', '.join(result.participants)}")

    return result


# Function calls to test step 2
# event_info = extract_event_info(
#     "Tomorrow I'm meeting Mark at the central plaza at 3pm."
# )
# parse_event_details(event_info.description)


def generate_confirmation(event_details: EventDetails) -> EventConfirmation:
    """Third LLM call to generate a confirmation message"""
    logger.info("Generating confirmation message")

    contents = types.Content(
        role="user", parts=[types.Part(text=str(event_details.model_dump()))]
    )

    config = types.GenerateContentConfig(
        system_instruction="Generate a natural confirmation message for the event. Sign of with your name; Susie",
        response_mime_type="application/json",
        response_schema=EventConfirmation,
    )

    completion = client.models.generate_content(
        model=model, config=config, contents=contents
    )

    result = completion.parsed

    assert isinstance(
        result, EventConfirmation
    ), f"Expected EventConfirmation, fot {type(result)}"

    logger.info("Confirmation message generated successfully")

    return result


# Function calls to test step 3
# event_info = extract_event_info(
#     "Tomorrow I'm meeting Mark at the central plaza at 3pm."
# )
# event_details = parse_event_details(event_info.description)
# confirmation_message = generate_confirmation(event_details)
# print(confirmation_message.confirmation_message)


# ----------------------------------------------------------
# Step 3: Chain the functions together
# ----------------------------------------------------------


def process_calendar_request(user_input: str) -> Optional[EventConfirmation]:
    """Main function implementing the prompt chain with gate check"""
    logger.info("Processing calendar request")
    logger.debug(f"Raw input: {user_input}")

    # First LLM call: Extract basic info
    initial_extraction = extract_event_info(user_input)

    # Gate check: Verifiy if it's a calendar event with sufficient confidence
    if (
        not initial_extraction.is_calendar_event
        or initial_extraction.confidence_score < 0.7
    ):
        logger.warning(
            f"Gate check failed - is_calender_event {initial_extraction.is_calendar_event}, confidence: {initial_extraction.confidence_score:.2f}"
        )
        return None

    logger.info("Gate check passed, proceeding with event processing")

    # Second LLM call: Get detailed event information
    event_details = parse_event_details(initial_extraction.description)

    # Third LLM call: Generate confirmation
    confirmation = generate_confirmation(event_details)

    logger.info("Calendar request processing completed successfully")
    return confirmation


# ----------------------------------------------------------
# Step 4: Test the chain with a valid input
# ----------------------------------------------------------

user_input = "Let's schedule a 1h team meeting next Tuesday at 2pm with Alice and Bob to discuss the project roadmap."

result = process_calendar_request(user_input)
if result:
    print(f"Confirmation: {result.confirmation_message}")
    if result.calendar_link:
        print(f"Calendar link: {result.calendar_link}")
else:
    print("This doesn't appear to be a calendar event request.")


# ----------------------------------------------------------
# Step 5: Test the chain with an invalid input
# ----------------------------------------------------------

user_input = "Can you send an email to Alice and Bob to discuss the project roadmap?"

result = process_calendar_request(user_input)
if result:
    print(f"Confirmation: {result.confirmation_message}")
    if result.calendar_link:
        print(f"Calendar link: {result.calendar_link}")
else:
    print("This doesn't appear to be a calendar event request.")
