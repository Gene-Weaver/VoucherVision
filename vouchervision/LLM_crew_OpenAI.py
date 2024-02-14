import os
from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI



class AIResearchCrew:
    def __init__(self, openai_api_key, OCR, JSON_rules, search_tool=None, llm=None):
        # Set the OPENAI API key
        os.environ["OPENAI_API_KEY"] = openai_api_key

        # Initialize the search tool, defaulting to DuckDuckGoSearchRun if not provided
        self.search_tool = search_tool if search_tool is not None else DuckDuckGoSearchRun()

        # Initialize the LLM (Language Learning Model), if provided
        self.llm = llm

        # Define the agents
        self.transcriber = Agent(
            role='Expert Text Parser',
            goal='Parse and rearrange unstructured OCR text into a standardized JSON dictionary',
            backstory="""You work at a museum transcribing specimen labels.
            Your expertise lies in precisely transcribing text and placing the text into the appropriate category.""",
            verbose=True,
            allow_delegation=False
            # Optionally include llm=self.llm here if an LLM was provided
        )

        self.spell_check = Agent(
            role='Spell Checker',
            goal='Correct any typos in the JSON key values',
            backstory="""Your job is to look at the JSON key values and use your knowledge to verify spelling. Your corrections should be incorporated into the JSON object that will be passed to the next employee, so return the spell-checked JSON dictionary or the previous JSON dictionary if no changes are required.""",
            verbose=True,
            allow_delegation=True,
            # Optionally include llm=self.llm here if an LLM was provided
        )

        self.fact_check = Agent(
            role='Fact Checker',
            goal='Verify the accuracy of taxonomy and location names',
            backstory="""Your job is to verify the plant taxonomy and geographic locations contained within the key values are accurate. You can use internet searches to check these fields. Your corrections should be incorporated into a new JSON object that will be passed to the next employee, so return the corrected JSON dictionary or the previous JSON dictionary if no changes are required.""",
            verbose=True,
            allow_delegation=True,
            tools=[self.search_tool]
            # Optionally include llm=self.llm here if an LLM was provided
        )

        self.validator = Agent(
            role='Synthesis',
            goal='Create a final museum JSON record',
            backstory="""You must produce a final JSON dictionary only.""",
            verbose=True,
            allow_delegation=True,
        )

        # Define the tasks
        self.task1 = Task(
            description=f"Use your knowledge to reformat, transform, and rearrange the unstructured text to fit the following requirements:{JSON_rules}. For null values, use an empty string. This is the unformatted OCR text: {OCR}",
            agent=self.transcriber
        )

        self.task2 = Task(
            description=f"The original text is OCR text, which may contain minor typos. Your job is to check all of the key values and fix any minor typos or spelling mistakes. You should remove any extraneous characters that should not belong in an official museum record.",
            agent=self.spell_check
        )
        
        self.task3 = Task(
            description="""Use your knowledge or search the internet to verify the information contained within the JSON dictionary. 
            For taxonomy, use the information contained in these keys: order, family, scientificName, scientificNameAuthorship, genus, specificEpithet, infraspecificEpithet.
            For geography, use the information contained in these keys: country, stateProvince, municipality, decimalLatitude, decimalLongitude.""",
            agent=self.fact_check
        )

        self.task4 = Task(
            description=f"Verify that the JSON dictionary is valid. If not, correct the error. Then print out the final JSON dictionary only without explanations.",
            agent=self.validator
        )

        # Create the crew
        # self.crew = Crew(
        #     agents=[self.transcriber, self.spell_check, self.fact_check, self.validator],
        #     tasks=[self.task1, self.task2, self.task3, self.task4],
        #     verbose=2,  # You can set it to 1 or 2 for different logging levels
        #     manager_llm=ChatOpenAI(temperature=0, model="gpt-4-1106-preview"),
        #     process=Process.hierarchical,
        # )
        self.crew = Crew(
            agents=[self.transcriber, self.validator],
            tasks=[self.task1, self.task4],
            manager_llm=ChatOpenAI(temperature=0, model="gpt-4-1106-preview"),
            process=Process.sequential,
            verbose=2  # You can set it to 1 or 2 for different logging levels
        )

    def execute_tasks(self):
        # Kick off the process and return the result
        result = self.crew.kickoff()
        print("######################")
        print(result)
        return result

if __name__ == "__main__":
    openai_api_key = ""
    OCR = "HERBARIUM OF MARYGROVE COLLEGE Name Carex scoparia V. condensa Fernald Locality Interlaken , Ind . Date 7/20/25 No ... ! Gerould Wilhelm & Laura Rericha \" Interlaken , \" was the site for many years of St. Joseph Novitiate , run by the Brothers of the Holy Cross . The buildings were on the west shore of Silver Lake , about 2 miles NE of Rolling Prairie , LaPorte Co. Indiana , ca. 41.688 \u00b0 N , 86.601 \u00b0 W Collector : Sister M. Vincent de Paul McGivney February 1 , 2011 THE UNIVERS Examined for the Flora of the Chicago Region OF 1817 MICH ! Ciscoparia SMVdeP University of Michigan Herbarium 1386297 copyright reserved cm Collector wortet 2010"
    JSON_rules = """This is the JSON template that includes instructions for each key
                {'catalogNumber': barcode identifier, at least 6 digits, fewer than 30 digits., 
                'order': full scientific name of the Order in which the taxon is classified. Order must be capitalized., 
                'family': full scientific name of the Family in which the taxon is classified. Family must be capitalized., 
                'scientificName': scientific name of the taxon including Genus, specific epithet, and any lower classifications., 
                'scientificNameAuthorship': authorship information for the scientificName formatted according to the conventions of the applicable Darwin Core nomenclaturalCode., 
                'genus': taxonomic determination to Genus, Genus must be capitalized., 
                'subgenus': name of the subgenus., 
                'specificEpithet': The name of the first or species epithet of the scientificName. Only include the species epithet., 
                'infraspecificEpithet': lowest or terminal infraspecific epithet of the scientificName., 
                'identifiedBy': list of names of people, doctors, professors, groups, or organizations who identified, determined the taxon name to the subject organism. This is not the specimen collector., recordedBy list of names of people, doctors, professors, groups, or organizations., 
                'recordNumber': identifier given to the specimen at the time it was recorded., 
                'verbatimEventDate': The verbatim original representation of the date and time information for when the specimen was collected., 
                'eventDate': collection date formatted as year-month-day YYYY-MM-DD., habitat habitat., 
                'occurrenceRemarks': all descriptive text in the OCR rearranged into sensible sentences or sentence fragments., 
                'country': country or major administrative unit., 
                'stateProvince': state, province, canton, department, region, etc., county county, shire, department, parish etc., 
                'municipality': city, municipality, etc., locality description of geographic information aiding in pinpointing the exact origin or location of the specimen., 
                'degreeOfEstablishment': cultivated plants are intentionally grown by humans. Use either - unknown or cultivated., 
                'decimalLatitude': latitude decimal coordinate., 
                'decimalLongitude': longitude decimal coordinate., verbatimCoordinates verbatim location coordinates., 
                'minimumElevationInMeters': minimum elevation or altitude in meters., 
                'maximumElevationInMeters': maximum elevation or altitude in meters.}"""
    ai_research_crew = AIResearchCrew(openai_api_key, OCR, JSON_rules)
    result = ai_research_crew.execute_tasks()