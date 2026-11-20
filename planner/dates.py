"""Deterministic flight/hotel alignment checks from the notebook."""
import json

def check_flight_hotel_dates(state):
    """Compare quoted hotel dates with destination-local arrivals from tool evidence."""
    calls = {}
    flight_results = []
    hotel_results = []
    for message in state.get('search_agent_messages', []):
        for call in getattr(message, 'tool_calls', []):
            calls[call['id']] = call
        call = calls.get(getattr(message, 'tool_call_id', None))
        if not call or not isinstance(message.content, str):
            continue
        try:
            payload = json.loads(message.content)
        except (ValueError, TypeError):
            continue
        if not isinstance(payload, dict) or payload.get('status') == 'error':
            continue
        if call['name'] == 'search_flights':
            flight_results.append(payload)
        elif call['name'] == 'search_hotels':
            hotel_results.append(payload)

    checks = []
    warnings = []
    for hotels in hotel_results:
        hotel_params = hotels.get('search_parameters', {})
        check_in = hotel_params.get('check_in_date')
        check_out = hotel_params.get('check_out_date')
        if not check_in or not check_out:
            continue
        for flights in flight_results:
            context = flights.get('live_web_context', {})
            for option in context.get('best_flights', []) + context.get('other_flights', []):
                legs = option.get('legs', [])
                if not legs:
                    continue
                arrival_time = legs[-1].get('arrival_time')
                if not arrival_time:
                    continue
                arrival_date = arrival_time[:10]
                comparison = {
                    'flight_numbers': [leg.get('flight_number') for leg in legs],
                    'destination': legs[-1].get('arrival_airport'),
                    'destination_local_arrival': arrival_time,
                    'requested_return_departure_date': flights.get('search_parameters', {}).get('return_date'),
                    'quoted_hotel_check_in': check_in,
                    'quoted_hotel_check_out': check_out,
                }
                if arrival_date >= check_out:
                    comparison['status'] = 'arrival_outside_hotel_stay'
                    warnings.append(
                        f'Flight arrival {arrival_date} falls on or after hotel checkout {check_out}; these options do not form a matching stay.'
                    )
                elif check_in < arrival_date:
                    comparison['status'] = 'hotel_starts_before_arrival'
                    warnings.append(
                        f'Flights arrive on {arrival_date}, but the quoted hotel stay starts {check_in}. '
                        'This includes a pre-arrival hotel night. Confirm whether it is intentional; '
                        f'otherwise re-search hotels from {arrival_date} to {check_out}. '
                        'Existing hotel prices must not be relabeled for the shorter stay.'
                    )
                elif check_in > arrival_date:
                    comparison['status'] = 'gap_before_hotel_check_in'
                    warnings.append(
                        f'Flight arrival {arrival_date} precedes hotel check-in {check_in}; accommodation for the gap is not covered.'
                    )
                else:
                    comparison['status'] = 'arrival_date_matches_check_in'
                checks.append(comparison)
    return {
        'option_checks': checks,
        'warnings': list(dict.fromkeys(warnings)),
        'limitations': [
            'Checks compare quoted search options, not a confirmed selected flight.',
            'Airport-to-hotel travel time and room access before normal check-in must be confirmed.',
            'Return-flight legs are not verified unless explicitly present in the tool results.',
        ],
    }

