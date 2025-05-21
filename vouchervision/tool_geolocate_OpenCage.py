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
        # Find the result with the highest confidence
        best_result_index = 0
        highest_confidence = 0
        
        for i, result in enumerate(results):
            # Check if confidence field exists and is higher than current highest
            if 'confidence' in result and result['confidence'] > highest_confidence:
                highest_confidence = result['confidence']
                best_result_index = i

        best_result = results[best_result_index]
        GEO_dict['GEO_method'] = 'OpenCageGeocode_forward'
        GEO_dict['GEO_formatted_full_string'] = best_result['formatted']
        GEO_dict['GEO_decimal_lat'] = best_result['geometry']['lat']
        GEO_dict['GEO_decimal_long'] = best_result['geometry']['lng']

        try:
            GEO_dict['GEO_city'] = best_result['components']['city']
        except:
            try:
                GEO_dict['GEO_city'] = best_result['components']['town']
            except:
                GEO_dict['GEO_city'] = ''

        try:
            GEO_dict['GEO_county'] = best_result['components']['county']
        except:
            try:
                GEO_dict['GEO_county'] = record.get('county', '').strip()
            except:
                GEO_dict['GEO_county'] = ''
        GEO_dict['GEO_state'] = best_result['components']['state']
        GEO_dict['GEO_state_code'] = best_result['components']['state_code']
        GEO_dict['GEO_country'] = best_result['components']['country']
        GEO_dict['GEO_country_code'] = best_result['components']['country_code']
        GEO_dict['GEO_continent'] = best_result['components']['continent']
    
    if GEO_dict['GEO_formatted_full_string'] and replace_if_success_geo:
        GEO_dict['GEO_override_OCR'] = True
        record['country'] = GEO_dict.get('GEO_country')
        record['stateProvince'] = GEO_dict.get('GEO_state')
        record['county'] = GEO_dict.get('GEO_county')
        record['municipality'] = GEO_dict.get('GEO_city')

    return record, GEO_dict




def main():
    import pandas as pd
    from tqdm import tqdm

    # Set OpenCage API key
    os.environ['OPENCAGE_API_KEY'] = "03332ffc6609417dac7bad144ce01054"

    # Read the CSV file
    input_file = 'D:/T_Downloads/Nesom_2025_revisted.xlsx'  # Change this to your input filename
    output_file = 'D:/T_Downloads/Nesom_2025_revisted_ww.xlsx'  # Change this to your desired output filename
    
    # Read CSV into pandas DataFrame
    df = pd.read_excel(input_file)
    
    # Ensure we have the required columns
    required_columns = ['State', 'County', 'Locality']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Required column '{col}' not found in CSV file.")
            return
    
    # Create a new DataFrame to store results
    results = []
    
    # Process each row
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Geocoding locations"):
        # Create a record dictionary with the renamed columns needed by validate_coordinates_opencage
        record = {
            'municipality': row['Locality'].strip() if pd.notna(row['Locality']) else '',
            'county': row['County'].strip() if pd.notna(row['County']) else '',
            'stateProvince': row['State'].strip() if pd.notna(row['State']) else '',
            'country': 'United States',  # Assuming all records are from US
            'decimalLatitude': '',
            'decimalLongitude': '',
            'verbatimCoordinates': ''
        }
        
        # Call the validation function
        validated_record, geo_info = validate_coordinates_opencage(record)
        
        # Combine the original row data with geocoding results
        result = row.to_dict()
        for key, value in geo_info.items():
            result[key] = value
        
        results.append(result)
    
    # Convert results to DataFrame and save to CSV
    result_df = pd.DataFrame(results)
    result_df.to_excel(output_file, index=False)
    print(f"Geocoding complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()