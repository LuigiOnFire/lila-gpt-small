# trainer/data_loader.py
import logging
from SPARQLWrapper import SPARQLWrapper, JSON
from trainer import config

class RDFDataFetcher:
    def __init__(self, sparql_endpoint, work_list_query, work_length_query):
        self.sparql_endpoint = sparql_endpoint
        self.work_list_query = work_list_query
        self.work_length_query = work_length_query
        self.sparql = SPARQLWrapper(self.sparql_endpoint)
        self.sparql.setReturnFormat(JSON)
        self.sparql.setTimeout(60)

    def _execute_query(self, current_query):
        self.sparql.setQuery(current_query)
        print(current_query)
        try:
            prior_results = self.sparql.query()
            print(prior_results)
            results = self.sparql.query().convert()
            return results["results"]["bindings"]
        except Exception as e:
            logging.error(f"SPARQL query failed: {e}")
            logging.error(f"Query: {current_query}")
            return []

    def fetch_work_names(self, limit=None, offset=None):
        """
        Fetches a list of works and their associated texts.
        Each item in the returned list is a dictionary, e.g.
        {'work_uri': 'https://dbpedia.org/resource/MyBook', 'text_content': 'This is an abstract.'}
        """
        query_to_execute = self.work_list_query

        # Apply limit and offset if provided and not already in the query string
        query_lower = query_to_execute.lower()
        if limit is not None and "limit" not in query_lower:
            query_to_execute += f" LIMIT {limit}"
        if offset is not None and "offset" not in query_lower:
            query_to_execute += f" OFFSET {offset}"

        logging.info(f"Fetching data with query\n{query_to_execute}")

        bindings = self._execute_query(query_to_execute)

        processed_results = []
        if not bindings:
            logging.warning("No results return for SPARQL query.")
            return processed_results

        for res in bindings:
            work_uri = res.get("work", {}).get("value")
            text_content = res.get("text", {}).get("value") # Assumes 'text' variable in SPARQL query
            
            if work_uri and text_content:
                processed_results.append({
                    "work_uri": work_uri,
                    "text_content": text_content
                })
            else:
                logging.warning(f"Missing 'work' or 'text' in results: {res}")

        logging.info(f"Fetched {len(processed_results)} items.")
        return processed_results

    def list_available_works_with_lengths(self, limit=100):
        """
        Placeholder for enumerating works and their lengths.
        This would typically require a more complex query or multiple queries
        depending on how 'length' is defined (characters, tokens, sentences)
        and if it's available in the KG.
        For now, it fetches texts and calculates character length.
        """
        items = self.fetch_work_names(limit=limit)
        works_with_lengths = []
        for item in items:
            works_with_lengths.append({
                "work_uri": item["work_uri"],
                "text_content": item["text_content"], # Keep for later use
                "char_length": len(item["text_content"])
            })
        # Sort by length for potential weighted sampling (descending) (why?)
        # work_with_lengths.sort(key=lambda x: x["char_length"], reverse=True)
        return works_with_lengths

# Example usage (can be in a notebook more main script)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    fetcher = RDFDataFetcher(config.SPARQL.ENDPOINT, config.SPARQL_QUERY_WORKS_AND_TEXTS)

    # 1. Fetch a list of texts (abstracts in this case)
    texts_data = fetcher.fetch_works_and_texts(limit=5)
    if texts_data:
        for item in texts_data:
            logging.info(f"Work URI: {item['work_uri']}, Text (snippet): {item['text_content'][:100]}...")

    else:
        logging.warning("No texts fetched.")

    # 2. Enumerate works and (character) lengths
    # works_info = fetcher.list_available_works_with_lengths(limit=5)
    # if works_info:
    #     for info in works_info:
    #         logging.info(f"Work: {info['work_uri]}, Char Length: {info['char_length']}")
    # else:
    #     logging.warning("No work info fetched.")
    
