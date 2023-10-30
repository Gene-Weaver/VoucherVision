"""
At the command line, only need to run once to install the package via pip:
$ pip install google-generativeai
"""

import google.generativeai as palm

palm.configure(api_key="YOUR API KEY")

defaults = {
  'model': 'models/text-bison-001',
  'temperature': 0,
  'candidate_count': 1,
  'top_k': 40,
  'top_p': 0.95,
  'max_output_tokens': 1024,
  'stop_sequences': [],
  'safety_settings': [{"category":"HARM_CATEGORY_DEROGATORY","threshold":1},{"category":"HARM_CATEGORY_TOXICITY","threshold":1},{"category":"HARM_CATEGORY_VIOLENCE","threshold":2},{"category":"HARM_CATEGORY_SEXUAL","threshold":2},{"category":"HARM_CATEGORY_MEDICAL","threshold":2},{"category":"HARM_CATEGORY_DANGEROUS","threshold":2}],
}
prompt = """1. Your job is to return a new dict based on the structure of the reference dict ref_dict and these are your rules. 
                    2. You must look at ref_dict and refactor the new text called OCR to match the same formatting. 
                    3. OCR contains unstructured text inside of [], use your knowledge to put the OCR text into the correct ref_dict column. 
                    4. If OCR is mostly empty and contains substantially less text than the ref_dict examples, then only return "None" and skip all other steps.
                    5. If there is a field that does not have a direct proxy in the OCR text, you can fill it in based on your knowledge, but you cannot generate new information.
                    6. Never put text from the ref_dict values into the new dict, but you must use the headers from ref_dict. 
                    7. There cannot be duplicate dictionary fields.
                    8. Only return the new dict, do not explain your answer.

    "Genus" - {"format": "[Genus]" or "[Family] indet" if no genus", "null_value": "", "description": taxonomic determination to genus, do captalize genus}
    "Species"- {"format": "[species]" or "indet" if no species, "null_value": "", "description": taxonomic determination to species, do not captalize species}
    "subspecies" - {"format": "[subspecies]", "null_value": "", "description": taxonomic determination to subspecies (subsp.)}
    "variety" - {"format": "[variety]", "null_value": "", "description": taxonomic determination to variety (var)}
    "forma" - {"format": "[form]", "null_value": "", "description": taxonomic determination to form (f.)}

    "Country" - {"format": "[Country]", "null_value": "no data", "description": Country that corresponds to the current geographic location of collection; capitalize first letter of each word; use the entire location name even if an abreviation is given}
    "State" - {"format": "[Adm. Division 1]", "null_value": "no data", "description": Administrative division 1 that corresponds to the current geographic location of collection; capitalize first letter of each word}
    "County" - {"format": "[Adm. Division 2]", "null_value": "no data", "description": Administrative division 2 that corresponds to the current geographic location of collection; capitalize first letter of each word}
    "Locality Name" - {"format": "verbatim", if no geographic info: "no data provided on label of catalog no: [######]", or if illegible: "locality present but illegible/not translated for catalog no: #######", or if no named locality: "no named locality for catalog no: #######", "description": "Description of geographic location or landscape"}

    "Min Elevation" - {format: "elevation integer", "null_value": "","description": Elevation or altitude in meters, convert from feet to meters if 'm' or 'meters' is not in the text and round to integer, default field for elevation if a range is not given}
    "Max Elevation" - {format: "elevation integer", "null_value": "","description": Elevation or altitude in meters, convert from feet to meters if 'm' or 'meters' is not in the text and round to integer, maximum elevation if there are two elevations listed but '' otherwise}
    "Elevation Units" - {format: "m", "null_value": "","description": "m" only if an elevation is present}
    
    "Verbatim Coordinates" - {"format": "[Lat, Long | UTM | TRS]", "null_value": "", "description": Verbatim coordinates as they appear on the label, fix typos to match standardized GPS coordinate format}

    "Datum" - {"format": "[WGS84, NAD23 etc.]", "null_value": "not present", "description": Datum of coordinates on label; "" is GPS coordinates are not in OCR}
    "Cultivated" - {"format": "yes", "null_value": "", "description": Indicates if specimen was grown in cultivation}
    "Habitat" - {"format": "verbatim", "null_value": "", "description": Description of habitat or location where specimen was collected, ignore descriptions of the plant itself}
    "Collectors" - {"format": "[Collector]", "null_value": "not present", "description": Full name of person (i.e., agent) who collected the specimen; if more than one person then separate the names with commas}
    "Collector Number" - {"format": "[Collector No.]", "null_value": "s.n.", "description": Sequential number assigned to collection, associated with the collector}
    "Verbatim Date" - {"format": "verbatim", "null_value": "s.d.", "description": Date of collection exactly as it appears on the label}
    "Date" - {"format": "[yyyy-mm-dd]", "null_value": "", "description": Date of collection formatted as year, month, and day; zeros may be used for unknown values i.e. 0000-00-00 if no date, YYYY-00-00 if only year, YYYY-MM-00 if no day}
    "End Date" - {"format": "[yyyy-mm-dd]", "null_value": "", "description": If date range is listed, later date of collection range}
input: El Kala   Algeria Aegilops El Tarf    1919-05-20 locality not transcribed for catalog no: 1702723  Charles d'Alleizette ovata May 20, 1919  s.n.    

output: {"Genus": "Aegilops", "Species": "ovata", "subspecies": "", "variety": "", "forma": "", "Country": "Algeria", "State": "El Tarf", "County": "El Kala", "Locality Name": "locality not transcribed for catalog no: 1702723", "Min Elevation": "", "Max Elevation": "", "Elevation Units": "", "Verbatim Coordinates": "", "Datum": "", "Cultivated": "", "Habitat": "", "Collectors": "Charles d'Alleizette", "Collector Number": "s.n.", "Verbatim Date": "May 20, 1919", "Date": "1919-05-20", "End Date": ""}

input: El Kala   Algeria Agrostis El Tarf    1918-06-08 locality not transcribed for catalog no: 1702919  Charles d'Alleizette pallida 8 Juin 1918  7748    

output: {"Genus": "Agrostis", "Species": "pallida", "subspecies": "", "variety": "", "forma": "", "Country": "Algeria", "State": "El Tarf", "County": "El Kala", "Locality Name": "locality not transcribed for catalog no: 1702919", "Min Elevation": "", "Max Elevation": "", "Elevation Units": "", "Verbatim Coordinates": "", "Datum": "", "Cultivated": "", "Habitat": "", "Collectors": "Charles d'Alleizette", "Collector Number": "7748", "Verbatim Date": "8 Juin 1918", "Date": "1918-06-08", "End Date": ""}

input: Gympie  river nr. sawmill Australia Hydrilla Queensland    1943-12-26 locality not transcribed for catalog no: 1702580  M. S. Clemens verticillata Dec. 26/43  43329    

output:"""

response = palm.generate_text(
  **defaults,
  prompt=prompt
)
print(response.result)