import requests
from urllib.parse import urlencode
from Levenshtein import ratio
from fuzzywuzzy import fuzz

class WFONameMatcher:
    def __init__(self, tool_WFO):
        self.base_url = "https://list.worldfloraonline.org/matching_rest.php?"
        self.N_BEST_CANDIDATES = 10
        self.NULL_DICT = {
                        "WFO_exact_match": False,
                        "WFO_exact_match_name": "",
                        "WFO_candidate_names": "",
                        "WFO_best_match": "",
                        "WFO_placement": "",
                        "WFO_override_OCR": False,
                    }
        self.SEP = '|'
        self.is_enabled = tool_WFO

    def extract_input_string(self, record):
        primary_input = f"{record.get('scientificName', '').strip()} {record.get('scientificNameAuthorship', '').strip()}".strip()
        secondary_input = ' '.join(filter(None, [record.get('genus', '').strip(), 
                                                 record.get('subgenus', '').strip(), 
                                                 record.get('specificEpithet', '').strip(), 
                                                 record.get('infraspecificEpithet', '').strip()])).strip()

        return primary_input, secondary_input

    def query_wfo_name_matching(self, input_string, check_homonyms=True, check_rank=True, accept_single_candidate=True):
        params = {
            "input_string": input_string,
            "check_homonyms": check_homonyms,
            "check_rank": check_rank,
            "method": "full",
            "accept_single_candidate": accept_single_candidate,
        }

        full_url = self.base_url + urlencode(params)

        response = requests.get(full_url)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": True, "message": "Failed to fetch data from WFO API"}
    
    def query_and_process(self, record):
        primary_input, secondary_input = self.extract_input_string(record)
        
        # Query with primary input
        primary_result = self.query_wfo_name_matching(primary_input)
        primary_processed, primary_ranked_candidates = self.process_wfo_response(primary_result, primary_input)

        if primary_processed.get('WFO_exact_match'):
            print("Selected Primary --- Exact Primary & Unchecked Secondary")
            return primary_processed
        else:
            # Query with secondary input
            secondary_result = self.query_wfo_name_matching(secondary_input)
            secondary_processed, secondary_ranked_candidates = self.process_wfo_response(secondary_result, secondary_input)

            if secondary_processed.get('WFO_exact_match'):
                print("Selected Secondary --- Unchecked Primary & Exact Secondary")
                return secondary_processed
            
            else:
                # Both failed, just return the first failure
                if (primary_processed.get("WFO_candidate_names") == '') and (secondary_processed.get("WFO_candidate_names") == ''):
                    print("Selected Primary --- Failed Primary & Failed Secondary")
                    return primary_processed
                
                # 1st failed, just return the second
                elif (primary_processed.get("WFO_candidate_names") == '') and (len(secondary_processed.get("WFO_candidate_names")) > 0):
                    print("Selected Secondary --- Failed Primary & Partial Secondary")
                    return secondary_processed
                
                # 2nd failed, just return the first
                elif (len(primary_processed.get("WFO_candidate_names")) > 0) and (secondary_processed.get("WFO_candidate_names") == ''):
                    print("Selected Primary --- Partial Primary & Failed Secondary")
                    return primary_processed

                # Both have partial matches, compare and rerank
                elif (len(primary_processed.get("WFO_candidate_names")) > 0) and (len(secondary_processed.get("WFO_candidate_names")) > 0):
                    # Combine and sort results, ensuring no duplicates
                    combined_candidates = list(set(primary_ranked_candidates + secondary_ranked_candidates))
                    combined_candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)  # Sort by similarity score, then name
                    
                    # Replace candidates with combined_candidates and combined best match 
                    best_score_primary = primary_processed["WFO_candidate_names"][0][1]
                    best_score_secondary = secondary_processed["WFO_candidate_names"][0][1]

                    # Extracting only the candidate names from the top candidates
                    top_candidates = combined_candidates[:self.N_BEST_CANDIDATES]
                    cleaned_candidates = [cand[0] for cand in top_candidates]

                    if best_score_primary >= best_score_secondary:
                            
                        primary_processed["WFO_candidate_names"] = cleaned_candidates
                        primary_processed["WFO_best_match"] = cleaned_candidates[0]
                        
                        response_placement = self.query_wfo_name_matching(primary_processed["WFO_best_match"])
                        placement_exact_match = response_placement.get("match")
                        primary_processed["WFO_placement"] = placement_exact_match.get("placement", '')
                        
                        print("Selected Primary --- Partial Primary & Partial Secondary")
                        return primary_processed
                    else:
                        secondary_processed["WFO_candidate_names"] = cleaned_candidates
                        secondary_processed["WFO_best_match"] = cleaned_candidates[0]

                        response_placement = self.query_wfo_name_matching(secondary_processed["WFO_best_match"])
                        placement_exact_match = response_placement.get("match")
                        secondary_processed["WFO_placement"] = placement_exact_match.get("placement", '')

                        print("Selected Secondary --- Partial Primary & Partial Secondary")
                        return secondary_processed
                else:
                    return self.NULL_DICT

    def process_wfo_response(self, response, query):
        simplified_response = {}
        ranked_candidates = None

        exact_match = response.get("match")
        simplified_response["WFO_exact_match"] = bool(exact_match)

        candidates = response.get("candidates", [])
        candidate_names = [candidate["full_name_plain"] for candidate in candidates] if candidates else []

        if not exact_match and candidate_names:
            cleaned_candidates, ranked_candidates = self._rank_candidates_by_similarity(query, candidate_names)
            simplified_response["WFO_candidate_names"] = cleaned_candidates
            simplified_response["WFO_best_match"] = cleaned_candidates[0] if cleaned_candidates else ''
        elif exact_match:
            simplified_response["WFO_candidate_names"] = exact_match.get("full_name_plain")
            simplified_response["WFO_best_match"] = exact_match.get("full_name_plain")
        else:
            simplified_response["WFO_candidate_names"] = ''
            simplified_response["WFO_best_match"] = ''

        # Call WFO again to update placement using WFO_best_match
        try:
            response_placement = self.query_wfo_name_matching(simplified_response["WFO_best_match"])
            placement_exact_match = response_placement.get("match")
            simplified_response["WFO_placement"] = placement_exact_match.get("placement", '')
        except:
            simplified_response["WFO_placement"] = ''

        return simplified_response, ranked_candidates
    
    def _rank_candidates_by_similarity(self, query, candidates):
        string_similarities = []
        fuzzy_similarities = {candidate: fuzz.ratio(query, candidate) for candidate in candidates}
        query_words = query.split()

        for candidate in candidates:
            candidate_words = candidate.split()
            # Calculate word similarities and sum them up
            word_similarities = [ratio(query_word, candidate_word) for query_word, candidate_word in zip(query_words, candidate_words)]
            total_word_similarity = sum(word_similarities)

            # Calculate combined similarity score (average of word and fuzzy similarities)
            fuzzy_similarity = fuzzy_similarities[candidate]
            combined_similarity = (total_word_similarity + fuzzy_similarity) / 2
            string_similarities.append((candidate, combined_similarity))

        # Sort the candidates based on combined similarity, higher scores first
        ranked_candidates = sorted(string_similarities, key=lambda x: x[1], reverse=True)

        # Extracting only the candidate names from the top candidates
        top_candidates = ranked_candidates[:self.N_BEST_CANDIDATES]
        cleaned_candidates = [cand[0] for cand in top_candidates]
        
        return cleaned_candidates, ranked_candidates
    
    def check_WFO(self, record, replace_if_success_wfo):
        if not self.is_enabled:
            return record, self.NULL_DICT

        else:
            self.replace_if_success_wfo = replace_if_success_wfo

            # "WFO_exact_match","WFO_exact_match_name","WFO_best_match","WFO_candidate_names","WFO_placement"
            simplified_response = self.query_and_process(record)
            simplified_response['WFO_override_OCR'] = False

            # best_match
            if simplified_response.get('WFO_exact_match'):
                simplified_response['WFO_exact_match_name'] = simplified_response.get('WFO_best_match')
            else:
                simplified_response['WFO_exact_match_name'] = ''

            # placement
            wfo_placement = simplified_response.get('WFO_placement', '')
            if wfo_placement:
                parts = wfo_placement.split('/')[1:]
                simplified_response['WFO_placement'] = self.SEP.join(parts)
            else:
                simplified_response['WFO_placement'] = ''

            if simplified_response.get('WFO_exact_match') and replace_if_success_wfo:
                simplified_response['WFO_override_OCR'] = True
                name_parts = simplified_response.get('WFO_placement').split('$')[0]
                name_parts = name_parts.split(self.SEP)
                record['order'] = name_parts[3]
                record['family'] = name_parts[4]
                record['genus'] = name_parts[5]
                record['specificEpithet'] = name_parts[6]
                record['scientificName'] = simplified_response.get('WFO_exact_match_name')

            return record, simplified_response
    
def validate_taxonomy_WFO(tool_WFO, record_dict, replace_if_success_wfo=False):
    Matcher = WFONameMatcher(tool_WFO)
    try:    
        record_dict, WFO_dict = Matcher.check_WFO(record_dict, replace_if_success_wfo)
        return record_dict, WFO_dict
    except:
        return record_dict, Matcher.NULL_DICT

'''
if __name__ == "__main__":
    Matcher = WFONameMatcher()
    # input_string = "Rhopalocarpus alterfolius"
    record_exact_match ={
        "order": "Malpighiales",
        "family": "Hypericaceae",
        "scientificName": "Hypericum prolificum",
        "scientificNameAuthorship": "",

        "genus": "Hypericum",
        "subgenus": "",
        "specificEpithet": "prolificum",
        "infraspecificEpithet": "",
    }
    record_partialPrimary_exactSecondary ={
        "order": "Malpighiales",
        "family": "Hypericaceae",
        "scientificName": "Hyperic prolificum",
        "scientificNameAuthorship": "",

        "genus": "Hypericum",
        "subgenus": "",
        "specificEpithet": "prolificum",
        "infraspecificEpithet": "",
    }
    record_exactPrimary_partialSecondary ={
        "order": "Malpighiales",
        "family": "Hypericaceae",
        "scientificName": "Hypericum prolificum",
        "scientificNameAuthorship": "",

        "genus": "Hyperic",
        "subgenus": "",
        "specificEpithet": "prolificum",
        "infraspecificEpithet": "",
    }
    record_partialPrimary_partialSecondary ={
        "order": "Malpighiales",
        "family": "Hypericaceae",
        "scientificName": "Hyperic prolificum",
        "scientificNameAuthorship": "",

        "genus": "Hypericum",
        "subgenus": "",
        "specificEpithet": "prolific",
        "infraspecificEpithet": "",
    }
    record_partialPrimary_partialSecondary_swap ={
        "order": "Malpighiales",
        "family": "Hypericaceae",
        "scientificName": "Hypericum prolific",
        "scientificNameAuthorship": "",

        "genus": "Hyperic",
        "subgenus": "",
        "specificEpithet": "prolificum",
        "infraspecificEpithet": "",
    }
    record_errorPrimary_partialSecondary ={
        "order": "Malpighiales",
        "family": "Hypericaceae",
        "scientificName": "ricum proli",
        "scientificNameAuthorship": "",

        "genus": "Hyperic",
        "subgenus": "",
        "specificEpithet": "prolificum",
        "infraspecificEpithet": "",
    }
    record_partialPrimary_errorSecondary ={
        "order": "Malpighiales",
        "family": "Hypericaceae",
        "scientificName": "Hyperic prolificum",
        "scientificNameAuthorship": "",

        "genus": "ricum",
        "subgenus": "",
        "specificEpithet": "proli",
        "infraspecificEpithet": "",
    }
    record_errorPrimary_errorSecondary ={
        "order": "Malpighiales",
        "family": "Hypericaceae",
        "scientificName": "ricum proli",
        "scientificNameAuthorship": "",

        "genus": "ricum",
        "subgenus": "",
        "specificEpithet": "proli",
        "infraspecificEpithet": "",
    }
    options = [record_exact_match,
               record_partialPrimary_exactSecondary,
               record_exactPrimary_partialSecondary,
               record_partialPrimary_partialSecondary,
               record_partialPrimary_partialSecondary_swap,
               record_errorPrimary_partialSecondary,
               record_partialPrimary_errorSecondary,
               record_errorPrimary_errorSecondary]
    for opt in options:
        simplified_response = Matcher.check_WFO(opt)
        print(json.dumps(simplified_response, indent=4))
'''