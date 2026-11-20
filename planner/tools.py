"""Flight and hotel search tools ported from the notebook."""
from .schemas import FlightSearchSchema, HotelSearchSchema

from langchain_core.tools import tool
from typing import Optional
from datetime import date
from .providers import GoogleSearch
import os

@tool(args_schema=FlightSearchSchema)
def search_flights(origin: str, destination: str, departure_date: date, return_date: Optional[date] = None, adults: int = 1, children: int = 0, two_way: bool = True, no_of_stops: int = 0) -> dict:
    """Search flight options with at most no_of_stops connections; count returned itineraries."""
    response = {
        'status': 'error',
        'number_of_flights': 0,
        'message': '',
        'search_parameters': {
            'origin': origin, 'destination': destination,
            'departure_date': str(departure_date),
            'return_date': str(return_date) if two_way and return_date else None,
            'passengers': {'adults': adults, 'children': children},
            'two_way': two_way, 'no_of_stops': no_of_stops
        },
        'live_web_context': {'best_flights': [], 'other_flights': []}
    }
    if two_way and return_date is None:
        response['message'] = 'A return_date is required for round-trip searches.'
        return response
    api_key = os.environ.get('SERPAPI_API_KEY')
    if not api_key:
        response['message'] = 'Set SERPAPI_API_KEY in your .env before searching.'
        return response

    search_params = {
        'engine': 'google_flights',
        'departure_id': origin, 'arrival_id': destination,
        'outbound_date': str(departure_date),
        'type': 1 if two_way else 2,
        'adults': adults, 'children': children,
        # SerpAPI: 0=any, 1=nonstop, 2=up to one stop, 3=up to two.
        'stops': no_of_stops + 1 if no_of_stops <= 2 else 0,
        'currency': 'USD', 'hl': 'en', 'api_key': api_key
    }
    if two_way:
        search_params['return_date'] = str(return_date)
    try:
        raw_results = GoogleSearch(search_params).get_dict()
    except Exception:
        response['message'] = 'Unable to fetch flights from SerpAPI. Check connectivity and API configuration.'
        return response
    provider_error = raw_results.get('error')
    no_results = (
        provider_error == "Google Flights hasn't returned any results for this query."
        and raw_results.get('search_metadata', {}).get('status') == 'Success'
    )
    if provider_error and not no_results:
        response['message'] = 'SerpAPI returned an error; flight availability could not be determined.'
        return response

    seen = set()
    for category in ('best_flights', 'other_flights'):
        for flight in raw_results.get(category, []):
            legs = flight.get('flights') or []
            if not legs:
                continue
            # Include intermediate stops hidden inside a single flight segment.
            stops_count = len(legs) - 1 + sum(
                len(leg.get('stops') or []) for leg in legs
            )
            if stops_count > no_of_stops:
                continue
            itinerary_legs = [{
                'airline': leg.get('airline'),
                'flight_number': leg.get('flight_number'),
                'departure_airport': leg.get('departure_airport', {}).get('id'),
                'arrival_airport': leg.get('arrival_airport', {}).get('id'),
                'departure_time': leg.get('departure_airport', {}).get('time'),
                'arrival_time': leg.get('arrival_airport', {}).get('time'),
                'duration_minutes': leg.get('duration'),
                'class': leg.get('travel_class')
            } for leg in legs]
            identity = (
                tuple((leg['flight_number'], leg['departure_airport'],
                       leg['arrival_airport'], leg['departure_time']) for leg in itinerary_legs),
                flight.get('price')
            )
            if identity in seen:
                continue
            seen.add(identity)
            response['live_web_context'][category].append({
                'total_duration_minutes': flight.get('total_duration'),
                'price': flight.get('price'), 'stops': stops_count,
                'legs': itinerary_legs
            })

    number_of_flights = sum(
        len(response['live_web_context'][category])
        for category in ('best_flights', 'other_flights')
    )
    response['number_of_flights'] = number_of_flights
    response['status'] = 'success' if number_of_flights else 'no_flights'
    stop_description = 'nonstop' if no_of_stops == 0 else f'with at most {no_of_stops} stop(s)'
    response['message'] = (
        f'Found {number_of_flights} flight option(s) {stop_description} from {origin} to {destination}.'
        if number_of_flights else
        f'No flights found {stop_description} from {origin} to {destination} for the requested dates.'
    )
    response['live_web_context']['price_insights'] = raw_results.get('price_insights', {})
    return response


import os
from datetime import date
from typing import Literal, Optional
from langchain_core.tools import tool
from .providers import GoogleSearch

@tool(args_schema=HotelSearchSchema)
def search_hotels(location: str, check_in_date: date, check_out_date: date,
                  adults: int = 2, children: int = 0, no_of_rooms: int = 1,
                  airbnb_type: bool = False, sort_by: Literal['price', 'rating', 'reviews'] = 'price',
                  children_ages: Optional[list[int]] = None) -> dict:
    """Search Google Hotels or vacation rentals; default to ascending nightly price in USD."""
    criteria = HotelSearchSchema(
        location=location, check_in_date=check_in_date, check_out_date=check_out_date,
        adults=adults, children=children, no_of_rooms=no_of_rooms,
        airbnb_type=airbnb_type, sort_by=sort_by, children_ages=children_ages
    )
    response = {
        'status': 'error', 'message': '', 'number_of_hotels': 0,
        'search_parameters': criteria.model_dump(mode='json'),
        'currency': 'USD', 'properties': [],
        'limitations': [
            'Room count cannot be filtered by the documented Google Hotels API.',
            'Prices and availability are provider quotes; confirm occupancy, taxes, and totals before booking.',
            'Counts and local sorting cover this result page only.'
        ]
    }
    if no_of_rooms != 1:
        response['message'] = 'Multi-room searches are unsupported by this API. Use no_of_rooms=1; multi-room availability and totals cannot be verified.'
        return response
    api_key = os.environ.get('SERPAPI_API_KEY')
    if not api_key:
        response['message'] = 'Set SERPAPI_API_KEY in your .env before searching.'
        return response
    params = {
        'engine': 'google_hotels', 'q': criteria.location,
        'check_in_date': str(criteria.check_in_date),
        'check_out_date': str(criteria.check_out_date),
        'adults': adults, 'children': children,
        'vacation_rentals': airbnb_type,
        'sort_by': {'price': 3, 'rating': 8, 'reviews': 13}[sort_by],
        'currency': 'USD', 'hl': 'en', 'api_key': api_key
    }
    if children:
        params['children_ages'] = ','.join(map(str, criteria.children_ages))
    try:
        raw = GoogleSearch(params).get_dict()
    except Exception:
        response['message'] = 'Unable to fetch hotels. Check connectivity and SerpAPI configuration.'
        return response
    if raw.get('error'):
        response['message'] = 'SerpAPI returned an error; hotel availability could not be determined.'
        return response

    properties = raw.get('properties') or []
    if not properties and raw.get('name') and raw.get('type') in ('hotel', 'vacation rental'):
        properties = [raw]  # Exact hotel-name searches may return property details.
    seen = set()
    for item in properties:
        identity = item.get('property_token') or (item.get('name'), item.get('link'))
        if identity in seen:
            continue
        seen.add(identity)
        nightly = item.get('rate_per_night') or {}
        total = item.get('total_rate') or {}
        response['properties'].append({
            'name': item.get('name'), 'type': item.get('type'),
            'description': item.get('description'), 'link': item.get('link'),
            'price_per_night': nightly.get('extracted_lowest'),
            'price_per_night_display': nightly.get('lowest'),
            'total_price': total.get('extracted_lowest'),
            'total_price_display': total.get('lowest'),
            'rating': item.get('overall_rating'), 'reviews': item.get('reviews'),
            'hotel_class': item.get('hotel_class'),
            'amenities': item.get('amenities') or [],
            'check_in_time': item.get('check_in_time'),
            'check_out_time': item.get('check_out_time')
        })
    def sort_key(item):
        field = {'price': 'price_per_night', 'rating': 'rating', 'reviews': 'reviews'}[sort_by]
        value = item.get(field)
        return (value is None, (value if sort_by == 'price' else -value) if value is not None else 0)
    response['properties'].sort(key=sort_key)
    response['number_of_hotels'] = len(response['properties'])
    response['status'] = 'success' if response['number_of_hotels'] else 'no_hotels'
    kind = 'vacation rental' if airbnb_type else 'hotel'
    response['message'] = (
        f"Found {response['number_of_hotels']} {kind} option(s) in {criteria.location}."
        if response['number_of_hotels'] else
        f'No {kind} options found in {criteria.location} for the requested dates.'
    )
    response['next_page_token'] = (raw.get('serpapi_pagination') or {}).get('next_page_token')
    return response
