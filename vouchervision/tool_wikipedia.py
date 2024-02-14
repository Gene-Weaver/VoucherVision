import itertools, wikipediaapi, requests, re, json
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
# from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun
import cProfile
import pstats

class WikipediaLinks():


    def __init__(self, json_file_path_wiki) -> None:
        self.json_file_path_wiki = json_file_path_wiki
        self.wiki_wiki = wikipediaapi.Wikipedia(
            user_agent='VoucherVision (merlin@example.com)',
            language='en'
        )
        self.property_to_rank = {
            'P225': 'Species',
            'P171': 'Family',
            'P105': 'Taxon rank',
            'P70': 'Genus',
            'P75': 'Clade',
            'P76': 'Subgenus',
            'P67': 'Subfamily',
            'P66': 'Tribe',
            'P71': 'Subtribe',
            'P61': 'Order',
            'P72': 'Suborder',
            'P73': 'Infraorder',
            'P74': 'Superfamily',
            'P142': 'Phylum',
            'P75': 'Clade',
            'P76': 'Subclass',
            'P77': 'Infraclass',
            'P78': 'Superorder',
            'P81': 'Class',
            'P82': 'Superclass',
            'P84': 'Kingdom',
            'P85': 'Superkingdom',
            'P86': 'Subkingdom',
            'P87': 'Infrakingdom',
            'P88': 'Parvkingdom',
            'P89': 'Domain',
            'P1421': 'GRIN',
            'P1070': 'KEW',
            'P5037': 'POWOID',
        }


    def get_label_for_entity_id(self, entity_id):
        url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbgetentities",
            "format": "json",
            "ids": entity_id,
            "props": "labels",
            "languages": "en"  # Assuming you want the label in English
        }
        response = requests.get(url, params=params)
        data = response.json()
        return data['entities'][entity_id]['labels']['en']['value'] if 'en' in data['entities'][entity_id]['labels'] else None


    def is_valid_url(self, url):
        try:
            response = requests.head(url, allow_redirects=True, timeout=5)
            # If the response status code is 200, the URL is reachable
            return response.status_code == 200
        except requests.RequestException as e:
            # If there was some issue with the request, such as the domain does not exist
            # print(f"URL {url} is not reachable. Error: {e}")
            return False
    
    # def get_infobar_data(self, wiki_page_title):
    #     # Step 1: Extract the Wikidata Item ID from the Wikipedia page
    #     wiki_api_url = "https://en.wikipedia.org/w/api.php"
    #     wiki_params = {
    #         "action": "query",
    #         "format": "json",
    #         "titles": wiki_page_title,
    #         "prop": "revisions",
    #         "rvprop": "content",
    #         "rvslots": "*"
    #     }

    #     wiki_response = requests.get(wiki_api_url, params=wiki_params)
    #     wiki_data = wiki_response.json()

    #     page_key = next(iter(wiki_data['query']['pages']))
    #     content = wiki_data['query']['pages'][page_key]['revisions'][0]['slots']['main']['*']

    #     infobox_pattern = re.compile(r'\{\{Infobox.*?\|title\}\}', re.DOTALL)    
    #     match = infobox_pattern.search(content)
    #     if match:
    #         wikidata_id =  match.group(1)  # Returns the full match including the 'Infobox' braces
    #     else:
    #         return "Infobox not found"

    #     # Step 2: Fetch Data from Wikidata Using the Extracted ID
    #     wikidata_api_url = "https://www.wikidata.org/w/api.php"
    #     wikidata_params = {
    #         "action": "wbgetentities",
    #         "format": "json",
    #         "ids": wikidata_id,
    #         "props": "claims"  # Adjust as needed to fetch the desired data
    #     }

    #     wikidata_response = requests.get(wikidata_api_url, params=wikidata_params)
    #     wikidata_content = wikidata_response.json()
        

    #     classification_full = {}
    #     classification = {}
    #     label_cache = {}  # Cache for labels


    #     # Turn this on to see the available properties to decode
    #     for prop_id, claims in wikidata_content['entities'][wikidata_id]['claims'].items():
    #         # Assuming the main snak value is what we want
    #         value = claims[0]['mainsnak']['datavalue']['value']
    #         if isinstance(value, dict):  # If the value is an entity ID
    #             # entity_id = value['id']
    #             # entity_id = value['id']
    #             if prop_id not in label_cache:
    #                 label_cache[prop_id] = self.get_label_for_entity_id(prop_id)
    #             classification_full[prop_id] = label_cache[prop_id]
    #         else:
    #             classification_full[prop_id] = value
    #     print(classification_full)
        # Map Wikidata properties to the corresponding taxonomic ranks

    def convert_to_decimal(self, coord_parts):
        lat_deg, lat_min, lat_dir, lon_deg, lon_min, lon_dir = coord_parts[:6]

        lat = float(lat_deg) + float(lat_min) / 60
        lon = float(lon_deg) + float(lon_min) / 60

        if lat_dir == 'S':
            lat = -lat
        if lon_dir == 'W':
            lon = -lon

        return f"{lat},{lon}"


    def extract_coordinates_and_region(self, coord_string):
        # Extract the coordinate parts and region info
        coord_parts = re.findall(r'(\d+|\w+)', coord_string)
        region_info = re.search(r'region:([^|]+)\|display', coord_string)

        if coord_parts and len(coord_parts) >= 6:
            # Convert to decimal coordinates
            decimal_coords = self.convert_to_decimal(coord_parts)
        else:
            decimal_coords = "Invalid coordinates format"

        region = region_info.group(1) if region_info else "Region not found"
        return decimal_coords, region
    

    def parse_infobox(self, infobox_string):
        # Split the string into lines
        lines = infobox_string.split('\n')

        # Dictionary to store the extracted data
        infobox_data = {}

        # Iterate over each line
        for line in lines:
            # Split the line into key and value
            parts = line.split('=', 1)

            # If the line is properly formatted with a key and value
            if len(parts) == 2:
                key = parts[0].strip()
                key = key.split(' ')[1]
                value = parts[1].strip()

                # Handling special cases like links or coordinates
                if value.startswith('[[') and value.endswith(']]'):
                    # Extracting linked article titles
                    value = value[2:-2].split('|')[0]
                elif value.startswith('{{coord') and value.endswith('}}'):
                    # Extracting coordinates
                    value = value[7:-2]
                elif value.startswith('[') and value.endswith(']') and ('http' in value):
                    value = value[1:-1]
                    url_parts = value.split(" ")
                    infobox_data['url_location'] = next((part for part in url_parts if 'http' in part), None)

                if key == 'coordinates':
                    decimal_coordinates, region = self.extract_coordinates_and_region(value)
                    infobox_data['region'] = region
                    infobox_data['decimal_coordinates'] = decimal_coordinates

                key = self.sanitize(key)
                value = self.sanitize(value)
                value = self.remove_html_and_wiki_markup(value)
                # Add to dictionary
                infobox_data[key] = value

        return infobox_data

    def get_infobox_data(self, wiki_page_title, opt=None):
        wiki_api_url = "https://en.wikipedia.org/w/api.php"
        wiki_params = {
            "action": "query",
            "format": "json",
            "titles": wiki_page_title,
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "*"
        }

        try:
            wiki_response = requests.get(wiki_api_url, params=wiki_params)
            wiki_response.raise_for_status()  # Check for HTTP errors
        except requests.RequestException as e:
            return f"Error fetching data: {e}"

        wiki_data = wiki_response.json()

        page_key = next(iter(wiki_data['query']['pages']), None)
        if page_key is None or "missing" in wiki_data['query']['pages'][page_key]:
            return "Page not found"

        content = wiki_data['query']['pages'][page_key]['revisions'][0]['slots']['main']['*']

        infobox_pattern = re.compile(r'\{\{Infobox.*?\}\}', re.DOTALL)
        match = infobox_pattern.search(content)
        
        if match:
            infobox_content = match.group()
        else:
            self.infobox_data = {}
            self.infobox_data_locality = {}
            return "Infobox not found"

        if opt is None:
            self.infobox_data = self.parse_infobox(infobox_content)
        else:
            self.infobox_data_locality = self.parse_infobox(infobox_content)



        # Example usage

        # for prop_id, claims in wikidata_content['entities'][wikidata_id]['claims'].items():
        #     # Get the taxonomic rank from the mapping
        #     rank = self.property_to_rank.get(prop_id)
        #     if rank:
        #         value = claims[0]['mainsnak']['datavalue']['value']
        #         if isinstance(value, dict):  # If the value is an entity ID
        #             entity_id = value['id']
        #             if entity_id not in label_cache:
        #                 label_cache[entity_id] = self.get_label_for_entity_id(entity_id)
        #             classification[rank] = label_cache[entity_id]
        #         else:
        #             classification[rank] = value

        # try:
        #     unknown_link = "https://powo.science.kew.org/taxon/" + classification['POWOID']
        #     if self.is_valid_url(unknown_link):
        #         classification['POWOID'] = unknown_link
        #         classification['POWOID_syn'] = unknown_link + '#synonyms'
        # except:
        #     pass
        # return classification



    def get_taxonbar_data(self, wiki_page_title):
        # Step 1: Extract the Wikidata Item ID from the Wikipedia page
        wiki_api_url = "https://en.wikipedia.org/w/api.php"
        wiki_params = {
            "action": "query",
            "format": "json",
            "titles": wiki_page_title,
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "*"
        }

        wiki_response = requests.get(wiki_api_url, params=wiki_params)
        wiki_data = wiki_response.json()

        page_key = next(iter(wiki_data['query']['pages']))
        content = wiki_data['query']['pages'][page_key]['revisions'][0]['slots']['main']['*']

        taxonbar_match = re.search(r'\{\{Taxonbar\|from=(Q\d+)\}\}', content)
        if not taxonbar_match:
            return "Taxonbar not found"

        wikidata_id = taxonbar_match.group(1)

        # Step 2: Fetch Data from Wikidata Using the Extracted ID
        wikidata_api_url = "https://www.wikidata.org/w/api.php"
        wikidata_params = {
            "action": "wbgetentities",
            "format": "json",
            "ids": wikidata_id,
            "props": "claims"  # Adjust as needed to fetch the desired data
        }

        wikidata_response = requests.get(wikidata_api_url, params=wikidata_params)
        wikidata_content = wikidata_response.json()
        

        classification_full = {}
        classification = {}
        label_cache = {}  # Cache for labels


        # Turn this on to see the available properties to decode
        # for prop_id, claims in wikidata_content['entities'][wikidata_id]['claims'].items():
        #     # Assuming the main snak value is what we want
        #     value = claims[0]['mainsnak']['datavalue']['value']
        #     if isinstance(value, dict):  # If the value is an entity ID
        #         # entity_id = value['id']
        #         # entity_id = value['id']
        #         if prop_id not in label_cache:
        #             label_cache[prop_id] = self.get_label_for_entity_id(prop_id)
        #         classification_full[prop_id] = label_cache[prop_id]
        #     else:
        #         classification_full[prop_id] = value
        # print(classification_full)
        # Map Wikidata properties to the corresponding taxonomic ranks
        

        for prop_id, claims in wikidata_content['entities'][wikidata_id]['claims'].items():
            # Get the taxonomic rank from the mapping
            rank = self.property_to_rank.get(prop_id)
            if rank:
                value = claims[0]['mainsnak']['datavalue']['value']
                if isinstance(value, dict):  # If the value is an entity ID
                    entity_id = value['id']
                    if entity_id not in label_cache:
                        label_cache[entity_id] = self.get_label_for_entity_id(entity_id)
                    classification[rank] = label_cache[entity_id]
                else:
                    classification[rank] = value

        try:
            unknown_link = "https://powo.science.kew.org/taxon/" + classification['POWOID']
            if self.is_valid_url(unknown_link):
                classification['POWOID'] = unknown_link
                classification['POWOID_syn'] = unknown_link + '#synonyms'
        except:
            pass
        return classification


    def extract_page_title(self, result_string):
        first_line = result_string.split('\n')[0]
        page_title = first_line.replace('Page: ', '').strip()
        return page_title


    def get_wikipedia_url(self, page_title):
        page = self.wiki_wiki.page(page_title)
        if page.exists():
            return page.fullurl
        else:
            return None


    def extract_info_taxa(self, page):
        links = []
        self.info_packet['WIKI_TAXA']['LINKS'] = {}
        self.info_packet['WIKI_TAXA']['DATA'] = {}

        self.info_packet['WIKI_TAXA']['DATA'].update(self.get_taxonbar_data(page.title))

        # for back in page.backlinks:
        #     back = self.sanitize(back) 
        #     if ':' not in back:
        #         link = self.sanitize(self.get_wikipedia_url(back))
        #         if link not in links:
        #             links.append(link)
        #             self.info_packet['WIKI_TAXA']['LINKS'][back] = link


    def extract_info_geo(self, page, opt=None):
        links = []
        self.info_packet['WIKI_GEO']['LINKS'] = {}
        if opt is None:
            self.get_infobox_data(page.title)
        else:
            self.get_infobox_data(page.title,opt=opt)

        for back in itertools.islice(page.backlinks, 10):  
            back = self.sanitize(back) 
            if ':' not in back:
                link = self.sanitize(self.get_wikipedia_url(back))
                if link not in links:
                    links.append(link)
                    self.info_packet['WIKI_GEO']['LINKS'][back] = link


    def gather_geo(self, query,opt=None):
        if opt is None:
            self.info_packet['WIKI_GEO']['DATA'] = {}
        else:
            self.info_packet['WIKI_LOCALITY']['DATA'] = {}
            
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

        result = wikipedia.run(query)
        summary = result.split('Summary:')[1]
        summary = self.sanitize(summary)
        # print(result)
        page_title = self.extract_page_title(result)

        page = self.wiki_wiki.page(page_title)

        # Do these first, they are less likely to fail
        if opt is None:
            self.info_packet['WIKI_GEO']['PAGE_LINK'] = self.get_wikipedia_url(page_title)
            self.info_packet['WIKI_GEO']['PAGE_TITLE'] = page_title
            self.info_packet['WIKI_GEO']['SUMMARY'] = summary

        else:
            self.info_packet['WIKI_LOCALITY']['PAGE_TITLE'] = page_title
            self.info_packet['WIKI_LOCALITY']['PAGE_LINK'] = self.get_wikipedia_url(page_title)
            self.info_packet['WIKI_LOCALITY']['SUMMARY'] = summary


        # Check if the page exists, get the more complex data. Do it last in case of failure ########################## This might not be useful enough to justify the time
        # if page.exists():
        #     if opt is None:
        #         self.extract_info_geo(page)
        #     else:
        #         self.extract_info_geo(page, opt=opt)

        if opt is None:
            self.info_packet['WIKI_GEO']['DATA'].update(self.infobox_data)
        else:
            self.info_packet['WIKI_LOCALITY']['DATA'].update(self.infobox_data_locality)


    def gather_taxonomy(self, query):
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

        # query = "Tracaulon sagittatum Tracaulon sagittatum"
        result = wikipedia.run(query)
        summary = result.split('Summary:')[1]
        summary = self.sanitize(summary)
        # print(result)
        page_title = self.extract_page_title(result)

        page = self.wiki_wiki.page(page_title)

        # Check if the page exists
        if page.exists():
            self.extract_info_taxa(page)

        self.info_packet['WIKI_TAXA']['PAGE_TITLE'] = page_title
        self.info_packet['WIKI_TAXA']['PAGE_LINK'] = self.get_wikipedia_url(page_title)
        self.info_packet['WIKI_TAXA']['SUMMARY'] = summary
        return self.info_packet 
    

    def gather_wikipedia_results(self, output):
        self.info_packet = {}
        self.info_packet['WIKI_TAXA'] = {}
        self.info_packet['WIKI_GEO'] = {}
        self.info_packet['WIKI_LOCALITY'] = {}

        municipality = output.get('municipality','')
        county = output.get('county','')
        stateProvince = output.get('stateProvince','')
        country = output.get('country','')

        locality = output.get('locality','')

        order = output.get('order','')
        family = output.get('family','')
        scientificName = output.get('scientificName','')
        genus = output.get('genus','')
        specificEpithet = output.get('specificEpithet','')


        query_geo = ' '.join([municipality, county, stateProvince, country]).strip()
        query_locality = locality.strip()
        query_taxa_primary = scientificName.strip()
        query_taxa_secondary = ' '.join([genus, specificEpithet]).strip()
        query_taxa_tertiary = ' '.join([order, family, genus, specificEpithet]).strip()

        # query_taxa = "Tracaulon sagittatum Tracaulon sagittatum"
        # query_geo = "Indiana Porter Co."
        # query_locality = "Mical Springs edge"
        
        if query_geo:
            try:
                self.gather_geo(query_geo)
            except:
                pass
        
        if query_locality:
            try:
                self.gather_geo(query_locality,'locality')
            except:
                pass
        
        queries_taxa = [query_taxa_primary, query_taxa_secondary, query_taxa_tertiary]
        for q in queries_taxa:
            if q:
                try:
                    self.gather_taxonomy(q)
                    break
                except:
                    pass

        # print(self.info_packet)
        # return self.info_packet
        # self.gather_geo(query_geo)
        try:
            with open(self.json_file_path_wiki, 'w', encoding='utf-8') as file:
                json.dump(self.info_packet, file, indent=4)
        except:
            sanitized_data = self.sanitize(self.info_packet)
            with open(self.json_file_path_wiki, 'w', encoding='utf-8') as file:
                json.dump(sanitized_data, file, indent=4)
        
        
    def sanitize(self, data):
        if isinstance(data, dict):
            return {self.sanitize(key): self.sanitize(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.sanitize(element) for element in data]
        elif isinstance(data, str):
            return data.encode('utf-8', 'ignore').decode('utf-8')
        else:
            return data
  
    def remove_html_and_wiki_markup(self, text):
        # Remove HTML tags
        clean_text = re.sub(r'<.*?>', '', text)

        # Remove Wiki links but keep the text inside
        # For example, '[[Greg Abbott]]' becomes 'Greg Abbott'
        clean_text = re.sub(r'\[\[(?:[^\]|]*\|)?([^\]|]*)\]\]', r'\1', clean_text)

        # Remove Wiki template markup, e.g., '{{nowrap|text}}' becomes 'text'
        clean_text = re.sub(r'\{\{(?:[^\}|]*\|)?([^\}|]*)\}\}', r'\1', clean_text)

        return clean_text


if __name__ == '__main__':
    test_output = {
    "filename": "MICH_7375774_Polygonaceae_Persicaria_",
    "catalogNumber": "1439649",
    "order": "",
    "family": "",
    "scientificName": "Tracaulon sagittatum",
    "scientificNameAuthorship": "",
    "genus": "Tracaulon",
    "subgenus": "",
    "specificEpithet": "sagittatum",
    "infraspecificEpithet": "",
    "identifiedBy": "",
    "recordedBy": "Marcus W. Lyon, Jr.",
    "recordNumber": "TX 11",
    "verbatimEventDate": "1927",
    "eventDate": "1927-00-00",
    "habitat": "wet subdunal woods",
    "occurrenceRemarks": "Flowers pink",
    "country": "Indiana",
    "stateProvince": "Porter Co.",
    "county": "",
    "municipality": "",
    "locality": "Mical Springs edge",
    "degreeOfEstablishment": "",
    "decimalLatitude": "",
    "decimalLongitude": "",
    "verbatimCoordinates": "",
    "minimumElevationInMeters": "",
    "maximumElevationInMeters": ""
    }
    do_print_profiler = True
    if do_print_profiler:
        profiler = cProfile.Profile()
        profiler.enable()
    
    Wiki = WikipediaLinks('D:/D_Desktop/usda_pdf/test.json')
    info_packet= Wiki.gather_wikipedia_results(test_output)

    if do_print_profiler:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumulative')
        stats.print_stats(50)
   