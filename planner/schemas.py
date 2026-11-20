"""Validated tool inputs, routing assignments, and shared graph state."""

"""
This module defines the validation schema for flight search parameters using Pydantic.
"""

from pydantic import BaseModel, Field, field_validator
from datetime import date
from typing import Optional
import re

class FlightSearchSchema(BaseModel):
    """
    Pydantic schema to strictly validate and normalize input parameters for the flight search tool.
    Checks for 3-letter IATA codes, valid passenger counts, and logical date ranges.
    """
    origin: str = Field(..., description="3-letter airport code (e.g., YYZ, LAX)")
    destination: str = Field(..., description="3-letter airport code (e.g., LHR, CDG)")
    departure_date: date = Field(..., description="Departure date in YYYY-MM-DD format")
    return_date: Optional[date] = Field(None, description="Optional return date in YYYY-MM-DD format")
    adults: int = Field(default=1, ge=1, description="Number of adult passengers (at least 1)")
    children: int = Field(default=0, ge=0, description="Number of child passengers (0 or more)")
    two_way: bool = Field(default=True, description="Boolean indicating a round trip journey")
    no_of_stops: int = Field(default=0, ge=0, description="Maximum number of stops allowed")

    @field_validator('origin', 'destination')
    @classmethod
    def validate_airport_code(cls, v: str) -> str:
        v_upper = v.strip().upper()
        if not re.match(r'^[A-Z]{3}$', v_upper):
            raise ValueError("Airport code must be exactly a 3-letter alphabetic code")
        return v_upper

    @field_validator('return_date')
    @classmethod
    def validate_return_date(cls, v: Optional[date], info) -> Optional[date]:
        dep_date = info.data.get('departure_date')
        if v and dep_date and v < dep_date:
            raise ValueError("Return date cannot be before the departure date")
        return v

from datetime import date
from typing import Annotated, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

class HotelSearchSchema(BaseModel):
    """Validate Google Hotels search criteria and guest ages."""
    location: str = Field(..., min_length=1, description='City, region, or hotel name')
    check_in_date: date = Field(..., description='Check-in date in YYYY-MM-DD format')
    check_out_date: date = Field(..., description='Check-out date; must be after check-in')
    adults: int = Field(default=2, ge=1)
    children: int = Field(default=0, ge=0)
    no_of_rooms: int = Field(default=1, ge=1, description='Requested rooms; API only supports this tool with one room')
    airbnb_type: bool = Field(default=False, description='True for vacation rentals, False for hotels; not Airbnb-exclusive')
    sort_by: Literal['price', 'rating', 'reviews'] = Field(default='reviews')
    children_ages: Optional[list[Annotated[int, Field(ge=1, le=17)]]] = Field(
        default=None, description='One age per child; use 1 for infants younger than one'
    )

    @field_validator('location')
    @classmethod
    def clean_location(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError('location must not be blank')
        return value

    @model_validator(mode='after')
    def validate_stay(self):
        if self.check_out_date <= self.check_in_date:
            raise ValueError('check_out_date must be after check_in_date')
        if len(self.children_ages or []) != self.children:
            raise ValueError('Provide one children_ages entry for each child')
        return self


from typing import Literal
from typing_extensions import TypedDict


class AgentTask(TypedDict):
    """One assignment passed to a travel workflow node.

    This describes the task input, rather than the full workflow state.
    """

    # The agent selected to handle this task.
    target_agent: Literal["search_agent", "travel_scout_agent"]
    # Preserve the user's original request and travel constraints.
    user_query: str
    # The specific assignment for the selected agent.
    focus: str


import operator
from typing import Annotated, List

from langchain_core.messages import BaseMessage


class TravelPlannerState(TypedDict):
    """Shared information passed between travel planner nodes.

    Fields annotated with operator.add accumulate lists. Nodes should return
    only new entries for those fields, rather than the entire existing list.
    Other fields are replaced when a node returns an update for them.
    TypedDict describes the schema; it does not create default values.
    """

    # Conversation history across the overall workflow.
    messages: Annotated[List[BaseMessage], operator.add]
    user_query: str

    # Assignments for the search agent and/or travel scout agent.
    tasks: List[AgentTask]
    require_synthesis: bool

    # Append one record per completed task, including its agent and answer.
    # Suggested record: {"target_agent": "...", "focus": "...", "answer": "..."}
    agent_results: Annotated[List[dict[str, str]], operator.add]
    synthesized_answer: str

    # Messages sent to each agent; planner refers to travel_scout_agent.
    search_agent_messages: Annotated[List[BaseMessage], operator.add]
    travel_scout_agent_msgs: Annotated[List[BaseMessage], operator.add]


from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator


class UserQueryClassifier(BaseModel):
    """Validated routing assignments produced by the classification LLM."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    tasks: List[AgentTask] = Field(
        ...,
        min_length=1,
        description=(
            "Assignments needed to answer the user request. Use search_agent "
            "for flights and hotels, and travel_scout_agent for destination "
            "research and itineraries. Preserve the user request in each task "
            "and give each task a specific focus."
        ),
    )
    require_synthesis: bool = Field(
        default=False,
        strict=True,
        description="Whether the agent findings need to be combined into one answer.",
    )

    @field_validator("tasks")
    @classmethod
    def validate_task_text(cls, tasks: List[AgentTask]) -> List[AgentTask]:
        validated_tasks = []
        for index, task in enumerate(tasks):
            cleaned_task = task.copy()
            for field_name in ("user_query", "focus"):
                value = task[field_name].strip()
                if not value:
                    raise ValueError(
                        f"tasks[{index}].{field_name} must not be blank"
                    )
                cleaned_task[field_name] = value
            validated_tasks.append(cleaned_task)
        return validated_tasks
