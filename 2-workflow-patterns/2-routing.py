import os
import logging

from typing import Optional, Literal
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


class CalendarRequestType(BaseModel):
    """Router LLM call: Determine the type of calendar request"""

    request_type: Literal["new_event", "modify_event", "other"] = Field(
        description="Type of calendar request being made"
    )
    confidence_score: float = Field(description="Confidence score between 0 and 1")
    description: str = Field(description="Cleaned description of the request")


class NewEventDetails(BaseModel):
    """Details for creating a new event"""

    name: str = Field(description="Name of the event")
    date: str = Field(description="Date and time of the event (ISO 8601)")
    duration_minutes: int = Field(description="Duration in minutes")
    participants: list[str] = Field(description="List of participants")


class Change(BaseModel):
    """Details for chaning an existing event"""

    field: str = Field(description="Field to change")
    new_value: str = Field(description="New value for the field")


class ModifyEventDetails(BaseModel):
    """Details for modifying an event"""

    event_identifier: str = Field(
        description="Description to identify the existing event"
    )
    changes: list[Change] = Field(description="List of changes to make")
    participants_to_add: list[str] = Field(description="New participants to add")
    participants_to_remove: list[str] = Field(description="Participants to remove")


class CalendarResponse(BaseModel):
    """Final response format"""

    success: bool = Field(description="Whether the operation was successful")
    message: str = Field(description="User-friendly response message")
    calendar_link: Optional[str] = Field(description="Calendar link if applicable")


# ----------------------------------------------------------
# Step 3 Define the routing and processing functions
# ----------------------------------------------------------


def route_calendar_request(user_input: str) -> CalendarRequestType:
    """Router LLM call to determine the type of calendar request"""
    logger.info("Routing calendar request")

    contents = types.Content(role="user", parts=[types.Part(text=user_input)])

    config = types.GenerateContentConfig(
        system_instruction="Determine if this is a request to create a new calendar event or modify an existing one.",
        response_mime_type="application/json",
        response_schema=CalendarRequestType,
    )

    completion = client.models.generate_content(
        model=model, config=config, contents=contents
    )

    result = completion.parsed

    assert isinstance(
        result, CalendarRequestType
    ), f"Expected CalendarRequestType, got {type(result)}"

    logger.info(
        f"Request routed as: {result.request_type} with confidence: {result.confidence_score}"
    )

    return result


def handle_new_event(description: str) -> CalendarResponse:
    """Process a new event request"""
    logger.info("Processing new event request")

    contents = types.Content(role="user", parts=[types.Part(text=description)])

    config = types.GenerateContentConfig(
        system_instruction="Extract details for creating a new calendar event.",
        response_mime_type="application/json",
        response_schema=NewEventDetails,
    )

    completion = client.models.generate_content(
        model=model, config=config, contents=contents
    )

    details = completion.parsed

    assert isinstance(
        details, NewEventDetails
    ), f"Expected NewEventDetails, got {type(details)}"

    logger.info(f"New event: {details.model_dump_json(indent=2)}")

    # Generate response
    return CalendarResponse(
        success=True,
        message=f"Created new event '{details.name}' for {details.date} with {', '.join(details.participants)}",
        calendar_link=f"calendar>//new?event={details.name}",
    )


def handle_modify_event(description: str) -> CalendarResponse:
    """Process an event modification request"""
    logger.info("Processing event modification request")

    # Get modification details
    contents = types.Content(role="user", parts=[types.Part(text=description)])

    config = types.GenerateContentConfig(
        system_instruction="Extract details for creating a new calendar event.",
        response_mime_type="application/json",
        response_schema=ModifyEventDetails,
    )

    completion = client.models.generate_content(
        model=model, config=config, contents=contents
    )

    details = completion.parsed

    assert isinstance(
        details, ModifyEventDetails
    ), f"Expected ModifyEventDetails, got {type(details)}"

    logger.info(f"New event: {details.model_dump_json(indent=2)}")

    return CalendarResponse(
        success=True,
        message=f"Modified event '{details.event_identifier}' with the requested changes",
        calendar_link=f"calendar://modify?event={details.event_identifier}",
    )


def process_calendar_request(user_input: str) -> Optional[CalendarResponse]:
    """Main function for implementing the routing workflow"""
    logger.info("Processing calendar request")

    # Route the request
    route_result = route_calendar_request(user_input)

    # Check confidence threshold
    if route_result.confidence_score < 0.7:
        logger.warning(f"Low confidence score: {route_result.confidence_score}")
        return None

    # Route to appropriate handlçer
    match route_result.request_type:
        case "new_event":
            return handle_new_event(route_result.description)
        case "modify_event":
            return handle_modify_event(route_result.description)
        case _:
            logger.warning("Request type not supported")
            return None


# --------------------------------------------------------------
# Step 3: Test with new event
# --------------------------------------------------------------

new_event_input = "Let's schedule a team meeting next Tuesday at 2pm with Alice and Bob"
result = process_calendar_request(new_event_input)
if result:
    print(f"Response: {result.message}")

# --------------------------------------------------------------
# Step 4: Test with modify event
# --------------------------------------------------------------

modify_event_input = (
    "Can you move the team meeting with Alice and Bob to Wednesday at 3pm instead?"
)
result = process_calendar_request(modify_event_input)
if result:
    print(f"Response: {result.message}")

# --------------------------------------------------------------
# Step 5: Test with invalid request
# --------------------------------------------------------------

invalid_input = "What's the weather like today?"
result = process_calendar_request(invalid_input)
if not result:
    print("Request not recognized as a calendar operation")
