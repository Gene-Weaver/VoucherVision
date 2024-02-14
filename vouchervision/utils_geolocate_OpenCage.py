import os
from opencage.geocoder import OpenCageGeocode
import pycountry_convert as pc
import warnings
import unicodedata
import pycountry_convert as pc
import warnings


### TODO 1/24/24
### If I want to use this instead of HERE, update the procedure for picking the best/most granular geolocation


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
    
def validate_coordinates_opencage(record, replace_if_success_geo=False):
    GEO_dict = {
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
    
    geocoder = OpenCageGeocode(os.environ['OPENCAGE_API_KEY'])

    query_loc = ', '.join(filter(None, [record.get('municipality', '').strip(), 
                                        record.get('county', '').strip(), 
                                        record.get('stateProvince', '').strip(), 
                                        record.get('country', '').strip()])).strip()
    
    
    query_decimal = ', '.join(filter(None, [record.get('decimalLatitude', '').strip(), 
                                        record.get('decimalLongitude', '').strip()])).strip()
    query_verbatim = record.get('verbatimCoordinates', '').strip()

    # results = geocoder.geocode('Ann Arbor, Michigan', no_annotations='1')
    results = geocoder.geocode(query_loc, no_annotations='1')

    if results:
        GEO_dict['GEO_method'] = 'OpenCageGeocode_forward'
        GEO_dict['GEO_formatted_full_string'] = results[0]['formatted']
        GEO_dict['GEO_decimal_lat'] = results[0]['geometry']['lat']
        GEO_dict['GEO_decimal_long'] = results[0]['geometry']['lng']

        GEO_dict['GEO_city'] = results[0]['components']['city']
        GEO_dict['GEO_county'] = results[0]['components']['county']
        GEO_dict['GEO_state'] = results[0]['components']['state']
        GEO_dict['GEO_state_code'] = results[0]['components']['state_code']
        GEO_dict['GEO_country'] = results[0]['components']['country']
        GEO_dict['GEO_country_code'] = results[0]['components']['country_code']
        GEO_dict['GEO_continent'] = results[0]['components']['continent']
    
    if GEO_dict['GEO_formatted_full_string'] and replace_if_success_geo:
        GEO_dict['GEO_override_OCR'] = True
        record['country'] = GEO_dict.get('GEO_country')
        record['stateProvince'] = GEO_dict.get('GEO_state')
        record['county'] = GEO_dict.get('GEO_county')
        record['municipality'] = GEO_dict.get('GEO_city')

    return record, GEO_dict


