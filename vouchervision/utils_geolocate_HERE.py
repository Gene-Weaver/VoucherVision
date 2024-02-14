import os, requests
import pycountry_convert as pc
import unicodedata
import pycountry_convert as pc
import warnings


def normalize_country_name(name):
    return unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')

def get_continent(country_name):
    warnings.filterwarnings("ignore", category=UserWarning, module='pycountry')

    continent_code_to_name = {
        "AF": "Africa",
        "NA": "North America",
        "OC": "Oceania",
        "AN": "Antarctica",
        "AS": "Asia",
        "EU": "Europe",
        "SA": "South America"
    }

    try:
        normalized_country_name = normalize_country_name(country_name)
        # Get country alpha2 code
        country_code = pc.country_name_to_country_alpha2(normalized_country_name)
        # Get continent code from country alpha2 code
        continent_code = pc.country_alpha2_to_continent_code(country_code)
        # Map the continent code to continent name
        return continent_code_to_name.get(continent_code, '')
    except Exception as e:
        print(str(e))
        return ''
    
def validate_coordinates_here(record, replace_if_success_geo=False):
    forward_url = 'https://geocode.search.hereapi.com/v1/geocode'
    reverse_url = 'https://revgeocode.search.hereapi.com/v1/revgeocode'
    
    pinpoint = ['GEO_city','GEO_county','GEO_state','GEO_country',]
    GEO_dict_null = {
        'GEO_override_OCR': False,
        'GEO_method': '',
        'GEO_formatted_full_string': '',
        'GEO_decimal_lat': '',
        'GEO_decimal_long': '',
        'GEO_city': '',
        'GEO_county': '',
        'GEO_state': '',
        'GEO_state_code': '',
        'GEO_country': '',
        'GEO_country_code': '',
        'GEO_continent': '',
    }
    GEO_dict = {
        'GEO_override_OCR': False,
        'GEO_method': '',
        'GEO_formatted_full_string': '',
        'GEO_decimal_lat': '',
        'GEO_decimal_long': '',
        'GEO_city': '',
        'GEO_county': '',
        'GEO_state': '',
        'GEO_state_code': '',
        'GEO_country': '',
        'GEO_country_code': '',
        'GEO_continent': '',
    }
    GEO_dict_rev = {
        'GEO_override_OCR': False,
        'GEO_method': '',
        'GEO_formatted_full_string': '',
        'GEO_decimal_lat': '',
        'GEO_decimal_long': '',
        'GEO_city': '',
        'GEO_county': '',
        'GEO_state': '',
        'GEO_state_code': '',
        'GEO_country': '',
        'GEO_country_code': '',
        'GEO_continent': '',
    }
    GEO_dict_rev_verbatim = {
        'GEO_override_OCR': False,
        'GEO_method': '',
        'GEO_formatted_full_string': '',
        'GEO_decimal_lat': '',
        'GEO_decimal_long': '',
        'GEO_city': '',
        'GEO_county': '',
        'GEO_state': '',
        'GEO_state_code': '',
        'GEO_country': '',
        'GEO_country_code': '',
        'GEO_continent': '',
    }
    GEO_dict_forward = {
        'GEO_override_OCR': False,
        'GEO_method': '',
        'GEO_formatted_full_string': '',
        'GEO_decimal_lat': '',
        'GEO_decimal_long': '',
        'GEO_city': '',
        'GEO_county': '',
        'GEO_state': '',
        'GEO_state_code': '',
        'GEO_country': '',
        'GEO_country_code': '',
        'GEO_continent': '',
    }
    GEO_dict_forward_locality = {
        'GEO_override_OCR': False,
        'GEO_method': '',
        'GEO_formatted_full_string': '',
        'GEO_decimal_lat': '',
        'GEO_decimal_long': '',
        'GEO_city': '',
        'GEO_county': '',
        'GEO_state': '',
        'GEO_state_code': '',
        'GEO_country': '',
        'GEO_country_code': '',
        'GEO_continent': '',
    }

    
    # For production
    query_forward = ', '.join(filter(None, [record.get('municipality', '').strip(),
                                        record.get('county', '').strip(),
                                        record.get('stateProvince', '').strip(),
                                        record.get('country', '').strip()])).strip()
    query_forward_locality = ', '.join(filter(None, [record.get('locality', '').strip(),
                                        record.get('municipality', '').strip(),
                                        record.get('county', '').strip(),
                                        record.get('stateProvince', '').strip(),
                                        record.get('country', '').strip()])).strip()
    query_reverse = ','.join(filter(None, [record.get('decimalLatitude', '').strip(),
                                        record.get('decimalLongitude', '').strip()])).strip()
    query_reverse_verbatim = record.get('verbatimCoordinates', '').strip()
    

    '''
    #For testing
    # query_forward = 'Ann bor, michign'
    query_forward = 'michigan'
    query_forward_locality = 'Ann bor, michign'
    # query_gps = "42 N,-83 W" # cannot have any spaces
    # query_reverse_verbatim = "42.278366,-83.744718" # cannot have any spaces
    query_reverse_verbatim = "42,-83" # cannot have any spaces
    query_reverse = "42,-83" # cannot have any spaces
    # params = {
    #     'q': query_loc,
    #     'apiKey': os.environ['HERE_API_KEY'],
    # }'''

    
    params_rev = {
        'at': query_reverse,
        'apiKey': os.environ['HERE_API_KEY'],
        'lang': 'en',
    }
    params_reverse_verbatim = {
        'at': query_reverse_verbatim,
        'apiKey': os.environ['HERE_API_KEY'],
        'lang': 'en',
    }
    params_forward = {
        'q': query_forward,
        'apiKey': os.environ['HERE_API_KEY'],
        'lang': 'en',
    }
    params_forward_locality = {
        'q': query_forward_locality,
        'apiKey': os.environ['HERE_API_KEY'],
        'lang': 'en',
    }

    ### REVERSE
    # If there are two string in the coordinates, try a reverse first based on the literal coordinates
    response = requests.get(reverse_url, params=params_rev)
    if response.status_code == 200:
        data = response.json()
        if data.get('items'):
            first_result = data['items'][0]
            GEO_dict_rev['GEO_method'] = 'HERE_Geocode_reverse'
            GEO_dict_rev['GEO_formatted_full_string'] = first_result.get('title', '')
            GEO_dict_rev['GEO_decimal_lat'] = first_result['position']['lat']
            GEO_dict_rev['GEO_decimal_long'] = first_result['position']['lng']

            address = first_result.get('address', {})
            GEO_dict_rev['GEO_city'] = address.get('city', '')
            GEO_dict_rev['GEO_county'] = address.get('county', '')
            GEO_dict_rev['GEO_state'] = address.get('state', '')
            GEO_dict_rev['GEO_state_code'] = address.get('stateCode', '')
            GEO_dict_rev['GEO_country'] = address.get('countryName', '')
            GEO_dict_rev['GEO_country_code'] = address.get('countryCode', '')
            GEO_dict_rev['GEO_continent'] = get_continent(address.get('countryName', ''))

    ### REVERSE Verbatim
    # If there are two string in the coordinates, try a reverse first based on the literal coordinates
    if GEO_dict_rev['GEO_city']: # If the reverse was successful, pass
        GEO_dict = GEO_dict_rev
    else:
        response = requests.get(reverse_url, params=params_reverse_verbatim)
        if response.status_code == 200:
            data = response.json()
            if data.get('items'):
                first_result = data['items'][0]
                GEO_dict_rev_verbatim['GEO_method'] = 'HERE_Geocode_reverse_verbatimCoordinates'
                GEO_dict_rev_verbatim['GEO_formatted_full_string'] = first_result.get('title', '')
                GEO_dict_rev_verbatim['GEO_decimal_lat'] = first_result['position']['lat']
                GEO_dict_rev_verbatim['GEO_decimal_long'] = first_result['position']['lng']

                address = first_result.get('address', {})
                GEO_dict_rev_verbatim['GEO_city'] = address.get('city', '')
                GEO_dict_rev_verbatim['GEO_county'] = address.get('county', '')
                GEO_dict_rev_verbatim['GEO_state'] = address.get('state', '')
                GEO_dict_rev_verbatim['GEO_state_code'] = address.get('stateCode', '')
                GEO_dict_rev_verbatim['GEO_country'] = address.get('countryName', '')
                GEO_dict_rev_verbatim['GEO_country_code'] = address.get('countryCode', '')
                GEO_dict_rev_verbatim['GEO_continent'] = get_continent(address.get('countryName', ''))

    ### FORWARD
    ### Try forward, if failes, try reverse using deci, then verbatim
    if GEO_dict_rev['GEO_city']: # If the reverse was successful, pass
        GEO_dict = GEO_dict_rev
    elif GEO_dict_rev_verbatim['GEO_city']:
        GEO_dict = GEO_dict_rev_verbatim
    else:
        response = requests.get(forward_url, params=params_forward)
        if response.status_code == 200:
            data = response.json()
            if data.get('items'):
                first_result = data['items'][0]
                GEO_dict_forward['GEO_method'] = 'HERE_Geocode_forward'
                GEO_dict_forward['GEO_formatted_full_string'] = first_result.get('title', '')
                GEO_dict_forward['GEO_decimal_lat'] = first_result['position']['lat']
                GEO_dict_forward['GEO_decimal_long'] = first_result['position']['lng']

                address = first_result.get('address', {})
                GEO_dict_forward['GEO_city'] = address.get('city', '')
                GEO_dict_forward['GEO_county'] = address.get('county', '')
                GEO_dict_forward['GEO_state'] = address.get('state', '')
                GEO_dict_forward['GEO_state_code'] = address.get('stateCode', '')
                GEO_dict_forward['GEO_country'] = address.get('countryName', '')
                GEO_dict_forward['GEO_country_code'] = address.get('countryCode', '')
                GEO_dict_forward['GEO_continent'] = get_continent(address.get('countryName', ''))

    ### FORWARD locality
    ### Try forward, if failes, try reverse using deci, then verbatim
    if GEO_dict_rev['GEO_city']: # If the reverse was successful, pass
        GEO_dict = GEO_dict_rev
    elif GEO_dict_rev_verbatim['GEO_city']:
        GEO_dict = GEO_dict_rev_verbatim
    elif GEO_dict_forward['GEO_city']:
        GEO_dict = GEO_dict_forward
    else:
        response = requests.get(forward_url, params=params_forward_locality)
        if response.status_code == 200:
            data = response.json()
            if data.get('items'):
                first_result = data['items'][0]
                GEO_dict_forward_locality['GEO_method'] = 'HERE_Geocode_forward_locality'
                GEO_dict_forward_locality['GEO_formatted_full_string'] = first_result.get('title', '')
                GEO_dict_forward_locality['GEO_decimal_lat'] = first_result['position']['lat']
                GEO_dict_forward_locality['GEO_decimal_long'] = first_result['position']['lng']

                address = first_result.get('address', {})
                GEO_dict_forward_locality['GEO_city'] = address.get('city', '')
                GEO_dict_forward_locality['GEO_county'] = address.get('county', '')
                GEO_dict_forward_locality['GEO_state'] = address.get('state', '')
                GEO_dict_forward_locality['GEO_state_code'] = address.get('stateCode', '')
                GEO_dict_forward_locality['GEO_country'] = address.get('countryName', '')
                GEO_dict_forward_locality['GEO_country_code'] = address.get('countryCode', '')
                GEO_dict_forward_locality['GEO_continent'] = get_continent(address.get('countryName', ''))

        
    # print(json.dumps(GEO_dict,indent=4))
    

    # Pick the most detailed version 
    # if GEO_dict_rev['GEO_formatted_full_string'] and GEO_dict_forward['GEO_formatted_full_string']:
    for loc in pinpoint:
        rev = GEO_dict_rev.get(loc,'')
        forward = GEO_dict_forward.get(loc,'')
        forward_locality = GEO_dict_forward_locality.get(loc,'')
        rev_verbatim = GEO_dict_rev_verbatim.get(loc,'')

        if not rev and not forward and not forward_locality and not rev_verbatim:
            pass
        elif rev:
            GEO_dict = GEO_dict_rev
            break
        elif forward:
            GEO_dict = GEO_dict_forward
            break
        elif forward_locality:
            GEO_dict = GEO_dict_forward_locality
            break
        elif rev_verbatim:
            GEO_dict = GEO_dict_rev_verbatim
            break
        else:
            GEO_dict = GEO_dict_null
            

    if GEO_dict['GEO_formatted_full_string'] and replace_if_success_geo:
        GEO_dict['GEO_override_OCR'] = True
        record['country'] = GEO_dict.get('GEO_country')
        record['stateProvince'] = GEO_dict.get('GEO_state')
        record['county'] = GEO_dict.get('GEO_county')
        record['municipality'] = GEO_dict.get('GEO_city')

    # print(json.dumps(GEO_dict,indent=4))
    return record, GEO_dict


if __name__ == "__main__":
    validate_coordinates_here(None)