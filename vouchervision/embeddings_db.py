import json, os, time, uuid
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from InstructorEmbedding import INSTRUCTOR
from langchain_community.vectorstores import Chroma
''' 
If there is a transformers install error:
pip install transformers==4.29.2
Python 3.8 and above will need to upgrade the transformers to 4.2x.xx
https://github.com/huggingface/transformers/issues/11799

The goal is to creat a domain knowledge database based on existing transcribed labels.

I modify the domain knowledge (an xlsx file) so that each row is embedded in a way that most closely
resembles the raw OCR output, since that is what will be used to query against the db.

Once the closest row is found, I use the id to go back to the xlsx and take the whole row, converting
it into a dictionary similar to the desired output from the LLM.

This dict is then added to the prompt as a hint for the LLM. 
'''

'''
pip uninstall protobuf
pip install protobuf==3.19.5 
'''
class VoucherVisionEmbedding:
    # def __init__(self, db_name, path_domain_knowledge, logger, build_new_db=False, model_name="hkunlp/instructor-xl", device="cuda"):
    #     DB_DIR = os.path.join(os.path.dirname(__file__), db_name)

    #     client_settings = chromadb.config.Settings(
    #         chroma_db_impl="duckdb+parquet",
    #         persist_directory=DB_DIR,
    #         anonymized_telemetry=False
    #     )
    #     embeddings = embedding_functions.InstructorEmbeddingFunction(model_name=model_name, device=device)

    #     self.collection = Chroma(
    #         collection_name="langchain_store",
    #         embedding_function=embeddings,
    #         client_settings=client_settings,
    #         persist_directory=DB_DIR,
    #     )

    #     total_rows = len(self.domain_knowledge)
    #     for index, row in self.domain_knowledge.iterrows():
    #         try:
    #             self.logger.info(f"[Creating New Embedding DB] --- Adding Row {index+1}/{total_rows}")
    #         except:
    #             print(f"Row {index+1}/{total_rows}")
    #         id = str(row[0])
    #         document = str(' '.join(row[1:][row[1:].notna()].astype(str)))

    #     self.collection.add_texts(document, None, id, embedding=embeddings)
    #     self.collection.persist()
    #     print(self.collection)

    def __init__(self, db_name, path_domain_knowledge, logger, build_new_db=False, model_name="hkunlp/instructor-xl", device="cuda"):
        DB_DIR = os.path.join(os.path.dirname(__file__), db_name)
        self.logger = logger
        self.path_domain_knowledge = path_domain_knowledge
        self.client = chromadb.PersistentClient(path=DB_DIR, 
                                                settings=Settings(anonymized_telemetry=False))
        
        ef = embedding_functions.InstructorEmbeddingFunction(model_name=model_name, device=device)
        self.domain_knowledge = pd.read_excel(path_domain_knowledge).fillna('').astype(str)
        
        if build_new_db:
            self.logger.info(f"Creating new DB from {self.path_domain_knowledge}")
            self.collection = self.client.create_collection(name=db_name, embedding_function=ef, metadata={"hnsw:space": "cosine"})
            self.create_db_from_xlsx()
        else:
            try:
                self.collection = self.client.get_collection(name=db_name, embedding_function=ef)
            except:
                self.logger.error(f"Embedding database not found! Creating new DB from {self.path_domain_knowledge}")
                self.collection = self.client.create_collection(name=db_name, embedding_function=ef, metadata={"hnsw:space": "cosine"})
                self.create_db_from_xlsx()


    def add_document(self, document, metadata, id):
        id = str(id)
        existing_documents = self.collection.get()
        if id not in existing_documents['ids']:
            try:
                self.collection.add(documents=[document], ids=[id])
            except Exception as e:
                self.logger.error(f"Error while adding document {id}: {str(e)}")


            # try:
            #     self.collection.add(documents=[document], ids=[id])
            # except:
            #     try:
            #         time.sleep(0.1)
            #         self.collection.add(documents=[document], ids=[id])
            #     except:
            #         try:
            #             self.logger.info(f"[Embedding Add Doc] --- Failed, skipping: {id}")
            #         except:
            #             print(f"Failed, skipping: {id}")
        else:
            try:
                self.logger.info(f"[Embedding Add Doc] --- ID already exists in the collection: {id}")
            except:
                print(f"ID already exists in the collection: {id}")


    def query_db(self, query_text, n_results):
        results = self.collection.query(query_texts=[query_text], n_results=n_results)

        self.similarity = round(results['distances'][0][0],3)
        self.similarity_exact = results['distances'][0][0]
        try:
            self.logger.info(f"[Embedding Search] --- Similarity (close to zero is best) {self.similarity}")
        except:
            print(f"Similarity (close to zero is best) --- {self.similarity}")

        self.domain_knowledge.iloc[:, 0] = self.domain_knowledge.iloc[:, 0].astype(str)
        
        # Initialize an empty list to hold dictionaries
        for id in results['ids']:
            row_dicts = self._get_row_from_df(id)
            if not row_dicts:
                # try:
                #     self.logger.info(f"[Embedding Search] --- Similar Dictionary\n{row_dicts}")
                # except:
                #     print(row_dicts)
            # else:
                try:
                    self.logger.info(f"[Embedding Search] --- No row found for id {id}")
                except:
                    print(f"No row found for id {id}")

        # Return the list of dictionaries if n_results > 1, else return single dictionary
        if n_results > 1:
            return row_dicts
        else:
            return row_dicts[0] if row_dicts else None

    def create_db_from_xlsx(self):
        total_rows = len(self.domain_knowledge)
        for index, row in self.domain_knowledge.iterrows():
            try:
                self.logger.info(f"[Creating New Embedding DB] --- Adding Row {index+1}/{total_rows}")
            except:
                print(f"Row {index+1}/{total_rows}")
            id = str(row.iloc[0])
            document = str(' '.join(row[0:][row[0:].notna()].astype(str)))
            self.add_document(document, None, id)

    def get_similarity(self):
        return self.similarity_exact
    
    def _get_row_from_df(self, ids):
        row_dicts = []  # initialize an empty list to hold dictionaries
        for id in ids:
            row = self.domain_knowledge[self.domain_knowledge.iloc[:, 0] == id]
            if not row.empty:
                row_dict = row.iloc[0].to_dict()
                row_dict.pop('Catalog Number', None)
                for key in row_dict:
                    if pd.isna(row_dict[key]):
                        row_dict[key] = ''
                row_dicts.append(row_dict)  # append the dictionary to the list
        return row_dicts if row_dicts else None  # return the list of dictionaries or None if it's empty

    # def _get_row_from_df(self, ids):
    #     for id in ids:
    #         row = self.domain_knowledge[self.domain_knowledge.iloc[:, 0] == id]
    #         if not row.empty:
    #             row_dict = row.iloc[0].to_dict()
    #             row_dict.pop('Catalog Number', None)
    #             for key in row_dict:
    #                 if pd.isna(row_dict[key]):
    #                     row_dict[key] = ''
    #             return row_dict
    #     return None
    



class VoucherVisionEmbeddingTest:
    def __init__(self, ground_truth_dir, llm_output_dir, model_name="hkunlp/instructor-xl"):
        self.ground_truth_dir = ground_truth_dir
        self.llm_output_dir = llm_output_dir
        self.model_name = model_name
        self.model = INSTRUCTOR(model_name, device="cuda")
        self.instruction = "Represent the Science json dictionary document:"

    def compare_texts(self, ground_truth_text, predicted_text):
        # Convert the texts to embeddings using the given model
        ground_truth_embedding = self.model.encode([[self.instruction,ground_truth_text]])
        predicted_embedding = self.model.encode([[self.instruction,predicted_text]])

        # Compute the cosine similarity between the two embeddings
        similarity = cosine_similarity(ground_truth_embedding, predicted_embedding)

        return similarity[0][0]
    
    @staticmethod
    def json_to_text(json_dict):
        return str(json_dict)
    
    def get_max_difference(self, similarities):
        differences = [abs(1 - sim) for sim in similarities]
        return max(differences)

    def evaluate(self):
        # Get a list of all ground truth and LLM output files
        ground_truth_files = os.listdir(self.ground_truth_dir)
        llm_output_files = os.listdir(self.llm_output_dir)

        # Ensure file lists are sorted so they match up correctly
        ground_truth_files.sort()
        llm_output_files.sort()

        similarities = []
        key_similarities = []  # List to store key similarity

        for ground_truth_file, llm_output_file in zip(ground_truth_files, llm_output_files):
            # Read the files and convert them to text
            with open(os.path.join(self.ground_truth_dir, ground_truth_file), 'r') as f:
                ground_truth_dict = json.load(f)
                ground_truth_text = self.json_to_text(ground_truth_dict)
            with open(os.path.join(self.llm_output_dir, llm_output_file), 'r') as ff:
                llm_output_dict = json.load(ff)
                llm_output_text = self.json_to_text(llm_output_dict)

            # Compute the similarity between the ground truth and the LLM output
            similarity = self.compare_texts(ground_truth_text, llm_output_text)

            # Clip and round to mitigate/smudge floating-point precision limitations
            similarity = np.clip(similarity, -1.0, 1.0)
            similarity = np.round(similarity, 6)

            similarities.append(similarity)

            # Compare keys
            ground_truth_keys = ', '.join(sorted(ground_truth_dict.keys()))
            llm_output_keys = ', '.join(sorted(llm_output_dict.keys()))
            key_similarity = self.compare_texts(ground_truth_keys, llm_output_keys)
            key_similarity = np.clip(key_similarity, -1.0, 1.0)
            key_similarity = np.round(key_similarity, 6)
            key_similarities.append(key_similarity)

        # Compute the mean similarity
        mean_similarity = np.mean(similarities)
        mean_key_similarity = np.mean(key_similarities)

        max_diff = self.get_max_difference(similarities)
        max_diff_key = self.get_max_difference(key_similarities)

        return mean_similarity, max_diff, similarities, mean_key_similarity, max_diff_key, key_similarities


    

if __name__ == '__main__':
    # db_name = "VV_all_asia_minimal"
    db_name = "all_asia_minimal"
    path_domain_knowledge = 'D:/Dropbox/LeafMachine2/leafmachine2/transcription/domain_knowledge/AllAsiaMinimalasof25May2023_2__FOR-EMBEDDING.xlsx'
    # path_domain_knowledge = 'D:/Dropbox/LeafMachine2/leafmachine2/transcription/domain_knowledge/AllAsiaMinimalasof25May2023_2__TRIMMEDtiny.xlsx'

    build_new_db = False


    VVE = VoucherVisionEmbedding(db_name, path_domain_knowledge, build_new_db)

    test_query = "Golden Thread\nHerbaria of Michigan State University (MSC) and\nUniversiti Kebangsaan Malaysia, Sabah Campus (UKMS)\nUNITED STATES\n3539788\nNATIONAL HERBARIUM\nPLANTS OF BORNEO\nBrookea tomentosa Benth.\nMalaysia. Sabah. Beaufort District: Beaufort Hill. 5°22'N,\n115°45'E. Elev. 200 m. Burned logged dipterocarp forest.\nCrocker Formation. Small tree, corolla cream.\nDet. at K, 1986\n28 August 1983\nWith: Reed S. Beaman and Teofila E. Beamann\nJohn H. Beaman 6844"

    domain_knowledge_example = VVE.query_db(test_query, 1)