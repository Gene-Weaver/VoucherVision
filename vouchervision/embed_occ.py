import openai
import os
import sys
import inspect
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import gradio as gr

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from vouchervision.general_utils import get_cfg_from_full_path
from prompts import PROMPT_UMICH_skeleton_all_asia
from vouchervision.LLM_OpenAI import num_tokens_from_string, OCR_to_dict

'''
This generates OpenAI embedding. These are no longer used by VoucherVision.
We have transitioned to "hkunlp/instructor-xl"

Please see: https://huggingface.co/hkunlp/instructor-xl

This file has  some experimentation code that can be helpful to reference, 
but is no relevant to VoucherVision.
'''

class GenerateEmbeddings:
    def __init__(self, file_occ, file_name, dir_out="D:/D_Desktop/embedding"):
        self.file_occ = file_occ
        self.file_name = file_name
        self.dir_out = dir_out

        self.SEP = '!!'

        # Set API key
        dir_home = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        path_cfg_private = os.path.join(dir_home, 'PRIVATE_DATA.yaml')
        cfg_private = get_cfg_from_full_path(path_cfg_private)
        openai.api_key = cfg_private['openai']['openai_api_key']

    def generate(self):
        # Read CSV file
        df = pd.read_csv(self.file_occ, sep='\t',
                         on_bad_lines='skip', dtype=str, low_memory=False)

        # Extract headers separately
        dwc_headers = df.columns.tolist()

        # Combine columns into a single string separated by commas
        df['combined'] = df.apply(
            lambda row: self.SEP.join(row.values.astype(str)), axis=1)

        # Wrap the get_embedding function call with tqdm progress bar
        tqdm.pandas(desc="Generating embeddings")
        df['ada_embedding'] = df.combined.progress_apply(
            lambda x: self.get_embedding(x, model='text-embedding-ada-002'))

        # Save to output CSV
        output_file = os.path.join(
            self.dir_out, f'embedded_dwc__{self.file_name}.csv')
        df[['combined', 'ada_embedding']].to_csv(output_file, index=False)

        # Save headers to a separate CSV file
        headers_file = os.path.join(
            self.dir_out, f'dwc_headers__{self.file_name}.csv')
        with open(headers_file, 'w') as f:
            f.write('\n'.join(dwc_headers))

        return output_file

    def get_embedding(self, text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

    def load_embedded_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        df['ada_embedding'] = df.ada_embedding.apply(eval).apply(np.array)

        headers_file = os.path.join(
            self.dir_out, f'dwc_headers__{self.file_name}.csv')
        with open(headers_file, 'r') as f:
            dwc_headers = f.read().splitlines()

        return df, dwc_headers

    def search_rows(self, dwc_headers, df, query, n=3, pprint=True):
        query_embedding = self.get_embedding(
            query, model="text-embedding-ada-002")
        df["similarity"] = df.ada_embedding.apply(
            lambda x: cosine_similarity([x], [query_embedding])[0][0])

        results = df.sort_values("similarity", ascending=False).head(n)

        if pprint:
            for i in range(n):
                row = results.iloc[i]
                df_split = pd.DataFrame(
                    [row.combined.split(self.SEP)], columns=dwc_headers)
                df_clean = df_split.replace(
                    'nan', np.nan).dropna(axis=1, how='any')
                # Convert df_clean to a dictionary
                row_dict = df_clean.to_dict(orient='records')[0]

                # Convert dictionary to a long literal string
                row_string = json.dumps(row_dict)

                print(row_string)
                # print(df_clean)
                # print(df_clean.to_string(index=False))

                nt = num_tokens_from_string(row_string, "cl100k_base")
                print(nt)

        return results


def create_embeddings(file_occ, file_name, dir_out):
    # Instantiate and generate embeddings
    embedder = GenerateEmbeddings(file_occ, file_name, dir_out)
    output_file = embedder.generate()


def old_method(img_path):
    set_rules = """1. Your job is to return a new dict based on the structure of the reference dict ref_dict and these are your rules. 
                2. You must look at ref_dict and refactor the new text called OCR to match the same formatting. 
                3. OCR contains unstructured text, use your knowledge to put the OCR text into the correct ref_dict column. 
                4. If there is a field that does not have a direct proxy in the OCR text, you can fill it in based on your knowledge, but you cannot generate new information.
                5. The dict key is the column header, the value is the new text. The separator in the new text is '!!', which indicates a new element but not strictly a new column. Remove the '!!' separator before adding text to the new dict
                6. Never put text from the ref_dict values into the new dict, but you must use the headers from ref_dict. 
                7. There cannot be duplicate dictionary fields.
                8. Only return the new dict, do not explain your answer."""
    
    # 4. If there is a simple typo you should correct the spelling, but do not rephrase or rework the ORC text.
    sample_text = """['gbifID', 'abstract', 'accessRights', 'accrualMethod', 'accrualPeriodicity', 'accrualPolicy', 'alternative', 'audience', 'available', 'bibliographicCitation', 'conformsTo', 'contributor', 'coverage', 'created', 'creator', 'date', 'dateAccepted', 'dateCopyrighted', 'dateSubmitted', 'description', 'educationLevel', 'extent', 'format', 'hasFormat', 'hasPart', 'hasVersion', 'identifier', 'instructionalMethod', 'isFormatOf', 'isPartOf', 'isReferencedBy', 'isReplacedBy', 'isRequiredBy', 'isVersionOf', 'issued', 'language', 'license', 'mediator', 'medium', 'modified', 'provenance', 'publisher', 'references', 'relation', 'replaces', 'requires', 'rights', 'rightsHolder', 'source', 'spatial', 'subject', 'tableOfContents', 'temporal', 'title', 'type', 'valid', 'institutionID', 'collectionID', 'datasetID', 'institutionCode', 'collectionCode', 'datasetName', 'ownerInstitutionCode', 'basisOfRecord', 'informationWithheld', 'dataGeneralizations', 'dynamicProperties', 'occurrenceID', 'catalogNumber', 'recordNumber', 'recordedBy', 'recordedByID', 'individualCount', 'organismQuantity', 'organismQuantityType', 'sex', 'lifeStage', 'reproductiveCondition', 'behavior', 'establishmentMeans', 'degreeOfEstablishment', 'pathway', 'georeferenceVerificationStatus', 'occurrenceStatus', 'preparations', 'disposition', 'associatedOccurrences', 'associatedReferences', 'associatedSequences', 'associatedTaxa', 'otherCatalogNumbers', 'occurrenceRemarks', 'organismID', 'organismName', 'organismScope', 'associatedOrganisms', 'previousIdentifications', 'organismRemarks', 'materialSampleID', 'eventID', 'parentEventID', 'fieldNumber', 'eventDate', 'eventTime', 'startDayOfYear', 'endDayOfYear', 'year', 'month', 'day', 'verbatimEventDate', 'habitat', 'samplingProtocol', 'sampleSizeValue', 'sampleSizeUnit', 'samplingEffort', 'fieldNotes', 'eventRemarks', 'locationID', 'higherGeographyID', 'higherGeography', 'continent', 'waterBody', 'islandGroup', 'island', 'countryCode', 'stateProvince', 'county', 'municipality', 'locality', 'verbatimLocality', 'verbatimElevation', 'verticalDatum', 'verbatimDepth', 'minimumDistanceAboveSurfaceInMeters', 'maximumDistanceAboveSurfaceInMeters', 'locationAccordingTo', 'locationRemarks', 'decimalLatitude', 'decimalLongitude', 'coordinateUncertaintyInMeters', 'coordinatePrecision', 'pointRadiusSpatialFit', 'verbatimCoordinateSystem', 'verbatimSRS', 'footprintWKT', 'footprintSRS', 'footprintSpatialFit', 'georeferencedBy', 'georeferencedDate', 'georeferenceProtocol', 'georeferenceSources', 'georeferenceRemarks', 'geologicalContextID', 'earliestEonOrLowestEonothem', 'latestEonOrHighestEonothem', 'earliestEraOrLowestErathem', 'latestEraOrHighestErathem', 'earliestPeriodOrLowestSystem', 'latestPeriodOrHighestSystem', 'earliestEpochOrLowestSeries', 'latestEpochOrHighestSeries', 'earliestAgeOrLowestStage', 'latestAgeOrHighestStage', 'lowestBiostratigraphicZone', 'highestBiostratigraphicZone', 'lithostratigraphicTerms', 'group', 'formation', 'member', 'bed', 'identificationID', 'verbatimIdentification', 'identificationQualifier', 'typeStatus', 'identifiedBy', 'identifiedByID', 'dateIdentified', 'identificationReferences', 'identificationVerificationStatus', 'identificationRemarks', 'taxonID', 'scientificNameID', 'acceptedNameUsageID', 'parentNameUsageID', 'originalNameUsageID', 'nameAccordingToID', 'namePublishedInID', 'taxonConceptID', 'scientificName', 'acceptedNameUsage', 'parentNameUsage', 'originalNameUsage', 'nameAccordingTo', 'namePublishedIn', 'namePublishedInYear', 'higherClassification', 'kingdom', 'phylum', 'class', 'order', 'family', 'subfamily', 'genus', 'genericName', 'subgenus', 'infragenericEpithet', 'specificEpithet', 'infraspecificEpithet', 'cultivarEpithet', 'taxonRank', 'verbatimTaxonRank', 'vernacularName', 'nomenclaturalCode', 'taxonomicStatus', 'nomenclaturalStatus', 'taxonRemarks', 'datasetKey', 'publishingCountry', 'lastInterpreted', 'elevation', 'elevationAccuracy', 'depth', 'depthAccuracy', 'distanceAboveSurface', 'distanceAboveSurfaceAccuracy', 'issue', 'mediaType', 'hasCoordinate', 'hasGeospatialIssues', 'taxonKey', 'acceptedTaxonKey', 'kingdomKey', 'phylumKey', 'classKey', 'orderKey', 'familyKey', 'genusKey', 'subgenusKey', 'speciesKey', 'species', 'acceptedScientificName', 'verbatimScientificName', 'typifiedName', 'protocol', 'lastParsed', 'lastCrawled', 'repatriated', 'relativeOrganismQuantity', 'level0Gid', 'level0Name', 'level1Gid', 'level1Name', 'level2Gid', 'level2Name', 'level3Gid', 'level3Name', 'iucnRedListCategory']\n3898509458,nan,http://rightsstatements.org/vocab/CNE/1.0/,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,2605588,nan,nan,nan,nan,nan,nan,nan,nan,nan,CC0_1_0,nan,nan,2022-08-15T08:30:45Z,nan,nan,https://portal.neherbaria.org/portal/collections/individual/index.php?occid=2605588,nan,nan,nan,nan,Mohonk Preserve,nan,nan,nan,nan,nan,nan,nan,nan,nan,745e5369-ba4e-4b80-b4b7-d64ab309e7b7,nan,Mohonk Preserve,DSRC,nan,nan,PRESERVED_SPECIMEN,nan,nan,nan,f2d1ba77-1c4d-41f6-8569-50becee5e9c3,MOH002237,nan,Dan Smiley,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,PRESENT,nan,nan,nan,nan,nan,nan,nan,The Buff,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,1971-10-21T00:00:00,nan,294,nan,1971,10,21,10/21/71,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,US,New York,nan,nan,Mohonk Lake,nan,nan,nan,nan,nan,nan,nan,nan,41.772115,-74.153723,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,830036,nan,nan,nan,nan,nan,nan,nan,Populus tremuloides Michx.,nan,nan,nan,nan,nan,nan,Plantae|Charophyta|Streptophytina|Equisetopsida|Magnoliidae|Malpighiales|Salicaceae|Populus,Plantae,Tracheophyta,Magnoliopsida,Malpighiales,Salicaceae,nan,Populus,Populus,nan,nan,tremuloides,nan,nan,SPECIES,nan,nan,nan,ACCEPTED,nan,nan,ffe1030d-42d1-4bb5-8400-1123cc859a5a,US,2022-11-29T23:03:56.952Z,nan,nan,nan,nan,nan,nan,GEODETIC_DATUM_ASSUMED_WGS84;AMBIGUOUS_COLLECTION;INSTITUTION_MATCH_FUZZY,StillImage,true,false,3040215,3040215,6,7707728,220,1414,6664,3040183,nan,3040215,Populus tremuloides,Populus tremuloides Michx.,Populus tremuloides,nan,DWC_ARCHIVE,2022-11-29T23:03:56.952Z,2022-11-29T23:02:54.980Z,false,nan,USA,United States,USA.33_1,New York,USA.33.57_1,Ulster,nan,nan,LC"""
    sample_text_headers = """['gbifID', 'abstract', 'accessRights', 'accrualMethod', 'accrualPeriodicity', 'accrualPolicy', 'alternative', 'audience', 'available', 'bibliographicCitation', 'conformsTo', 'contributor', 'coverage', 'created', 'creator', 'date', 'dateAccepted', 'dateCopyrighted', 'dateSubmitted', 'description', 'educationLevel', 'extent', 'format', 'hasFormat', 'hasPart', 'hasVersion', 'identifier', 'instructionalMethod', 'isFormatOf', 'isPartOf', 'isReferencedBy', 'isReplacedBy', 'isRequiredBy', 'isVersionOf', 'issued', 'language', 'license', 'mediator', 'medium', 'modified', 'provenance', 'publisher', 'references', 'relation', 'replaces', 'requires', 'rights', 'rightsHolder', 'source', 'spatial', 'subject', 'tableOfContents', 'temporal', 'title', 'type', 'valid', 'institutionID', 'collectionID', 'datasetID', 'institutionCode', 'collectionCode', 'datasetName', 'ownerInstitutionCode', 'basisOfRecord', 'informationWithheld', 'dataGeneralizations', 'dynamicProperties', 'occurrenceID', 'catalogNumber', 'recordNumber', 'recordedBy', 'recordedByID', 'individualCount', 'organismQuantity', 'organismQuantityType', 'sex', 'lifeStage', 'reproductiveCondition', 'behavior', 'establishmentMeans', 'degreeOfEstablishment', 'pathway', 'georeferenceVerificationStatus', 'occurrenceStatus', 'preparations', 'disposition', 'associatedOccurrences', 'associatedReferences', 'associatedSequences', 'associatedTaxa', 'otherCatalogNumbers', 'occurrenceRemarks', 'organismID', 'organismName', 'organismScope', 'associatedOrganisms', 'previousIdentifications', 'organismRemarks', 'materialSampleID', 'eventID', 'parentEventID', 'fieldNumber', 'eventDate', 'eventTime', 'startDayOfYear', 'endDayOfYear', 'year', 'month', 'day', 'verbatimEventDate', 'habitat', 'samplingProtocol', 'sampleSizeValue', 'sampleSizeUnit', 'samplingEffort', 'fieldNotes', 'eventRemarks', 'locationID', 'higherGeographyID', 'higherGeography', 'continent', 'waterBody', 'islandGroup', 'island', 'countryCode', 'stateProvince', 'county', 'municipality', 'locality', 'verbatimLocality', 'verbatimElevation', 'verticalDatum', 'verbatimDepth', 'minimumDistanceAboveSurfaceInMeters', 'maximumDistanceAboveSurfaceInMeters', 'locationAccordingTo', 'locationRemarks', 'decimalLatitude', 'decimalLongitude', 'coordinateUncertaintyInMeters', 'coordinatePrecision', 'pointRadiusSpatialFit', 'verbatimCoordinateSystem', 'verbatimSRS', 'footprintWKT', 'footprintSRS', 'footprintSpatialFit', 'georeferencedBy', 'georeferencedDate', 'georeferenceProtocol', 'georeferenceSources', 'georeferenceRemarks', 'geologicalContextID', 'earliestEonOrLowestEonothem', 'latestEonOrHighestEonothem', 'earliestEraOrLowestErathem', 'latestEraOrHighestErathem', 'earliestPeriodOrLowestSystem', 'latestPeriodOrHighestSystem', 'earliestEpochOrLowestSeries', 'latestEpochOrHighestSeries', 'earliestAgeOrLowestStage', 'latestAgeOrHighestStage', 'lowestBiostratigraphicZone', 'highestBiostratigraphicZone', 'lithostratigraphicTerms', 'group', 'formation', 'member', 'bed', 'identificationID', 'verbatimIdentification', 'identificationQualifier', 'typeStatus', 'identifiedBy', 'identifiedByID', 'dateIdentified', 'identificationReferences', 'identificationVerificationStatus', 'identificationRemarks', 'taxonID', 'scientificNameID', 'acceptedNameUsageID', 'parentNameUsageID', 'originalNameUsageID', 'nameAccordingToID', 'namePublishedInID', 'taxonConceptID', 'scientificName', 'acceptedNameUsage', 'parentNameUsage', 'originalNameUsage', 'nameAccordingTo', 'namePublishedIn', 'namePublishedInYear', 'higherClassification', 'kingdom', 'phylum', 'class', 'order', 'family', 'subfamily', 'genus', 'genericName', 'subgenus', 'infragenericEpithet', 'specificEpithet', 'infraspecificEpithet', 'cultivarEpithet', 'taxonRank', 'verbatimTaxonRank', 'vernacularName', 'nomenclaturalCode', 'taxonomicStatus', 'nomenclaturalStatus', 'taxonRemarks', 'datasetKey', 'publishingCountry', 'lastInterpreted', 'elevation', 'elevationAccuracy', 'depth', 'depthAccuracy', 'distanceAboveSurface', 'distanceAboveSurfaceAccuracy', 'issue', 'mediaType', 'hasCoordinate', 'hasGeospatialIssues', 'taxonKey', 'acceptedTaxonKey', 'kingdomKey', 'phylumKey', 'classKey', 'orderKey', 'familyKey', 'genusKey', 'subgenusKey', 'speciesKey', 'species', 'acceptedScientificName', 'verbatimScientificName', 'typifiedName', 'protocol', 'lastParsed', 'lastCrawled', 'repatriated', 'relativeOrganismQuantity', 'level0Gid', 'level0Name', 'level1Gid', 'level1Name', 'level2Gid', 'level2Name', 'level3Gid', 'level3Name', 'iucnRedListCategory']"""

    sample_OCR_response = """PLANTS OF BORNEC!! Euphorbiaceae!! Chaetocarpus castanocarpus Thwaites!! Det. JH Beaman, 15 May 2010 !!Sabah: Kota Kinabalu District: Bukit Padang, by UKMS!!temporary campus. Elev. 30 m. Eroded hills and gullies.!!scattered scrubby vegetation; Crocker Formation. Shrub.!!Lat. 5°58 N. Long. 116°06 E!!John H. Beaman 83041!!August 1983!!with Willem Meijer!!HERBARIA OF UNIVERSITI KEBANGSAAN MALAYSIA (UKMS) and!!MICHIGAN STATE UNIVERSITY (MSC)!!"""
    sample_dict = """{"gbifID": "3898509458", "accessRights": "http://rightsstatements.org/vocab/CNE/1.0/", "identifier": "2605588", "license": "CC0_1_0", "modified": "2022-08-15T08:30:45Z", "references": "https://portal.neherbaria.org/portal/collections/individual/index.php?occid=2605588", "rightsHolder": "Mohonk Preserve", "collectionID": "745e5369-ba4e-4b80-b4b7-d64ab309e7b7", "institutionCode": "Mohonk Preserve", "collectionCode": "DSRC", "basisOfRecord": "PRESERVED_SPECIMEN", "occurrenceID": "f2d1ba77-1c4d-41f6-8569-50becee5e9c3", "catalogNumber": "MOH002237", "recordedBy": "Dan Smiley", "occurrenceStatus": "PRESENT", "occurrenceRemarks": "The Buff", "eventDate": "1971-10-21T00:00:00", "startDayOfYear": "294", "year": "1971", "month": "10", "day": "21", "verbatimEventDate": "10/21/71", "countryCode": "US", "stateProvince": "New York", "locality": "Mohonk Lake", "decimalLatitude": "41.772115", "decimalLongitude": "-74.153723", "taxonID": "830036", "scientificName": "Populus tremuloides Michx.", "higherClassification": "Plantae|Charophyta|Streptophytina|Equisetopsida|Magnoliidae|Malpighiales|Salicaceae|Populus", "kingdom": "Plantae", "phylum": "Tracheophyta", "class": "Magnoliopsida", "order": "Malpighiales", "family": "Salicaceae", "genus": "Populus", "genericName": "Populus", "specificEpithet": "tremuloides", "taxonRank": "SPECIES", "taxonomicStatus": "ACCEPTED", "datasetKey": "ffe1030d-42d1-4bb5-8400-1123cc859a5a", "publishingCountry": "US", "lastInterpreted": "2022-11-29T23:03:56.952Z", "issue": "GEODETIC_DATUM_ASSUMED_WGS84;AMBIGUOUS_COLLECTION;INSTITUTION_MATCH_FUZZY", "mediaType": "StillImage", "hasCoordinate": "true", "hasGeospatialIssues": "false", "taxonKey": "3040215", "acceptedTaxonKey": "3040215", "kingdomKey": "6", "phylumKey": "7707728", "classKey": "220", "orderKey": "1414", "familyKey": "6664", "genusKey": "3040183", "speciesKey": "3040215", "species": "Populus tremuloides", "acceptedScientificName": "Populus tremuloides Michx.", "verbatimScientificName": "Populus tremuloides", "protocol": "DWC_ARCHIVE", "lastParsed": "2022-11-29T23:03:56.952Z", "lastCrawled": "2022-11-29T23:02:54.980Z", "repatriated": "false", "level0Gid": "USA", "level0Name": "United States", "level1Gid": "USA.33_1", "level1Name": "New York", "level2Gid": "USA.33.57_1", "level2Name": "Ulster", "iucnRedListCategory": "LC"}"""

    nt_rules = num_tokens_from_string(set_rules, "cl100k_base")
    nt_dict = num_tokens_from_string(sample_dict, "cl100k_base")
    nt_ocr = num_tokens_from_string(sample_OCR_response, "cl100k_base")

    print(f"nt - nt_rules {nt_rules}")
    print(f"nt - nt_dict {nt_dict}")
    print(f"nt - nt_new {nt_ocr}")

    do_create = False

    file_occ = 'D:/Dropbox/LeafMachine2/leafmachine2/transcription/test_occ/occurrence_short.txt'
    file_name = 'test_occ'
    dir_out = "D:/D_Desktop/embedding"

    '''
    if do_create:
        create_embeddings(file_occ, file_name, dir_out)
    

    # Load the generated embeddings
    output_file = os.path.join(dir_out, f'embedded_dwc__{file_name}.csv')
    embedder = GenerateEmbeddings(file_occ, file_name, dir_out)
    embedded_df, dwc_headers = embedder.load_embedded_csv(output_file)

    # Search for reviews
    search_query = "1971 The Buff"
    results = embedder.search_rows(dwc_headers, embedded_df, search_query, n=1)
    print(results)
    '''
    GPT_response = OCR_to_dict(img_path)
    print(GPT_response)


if __name__ == '__main__':
    print()

