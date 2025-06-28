import requests

query = """
PREFIX co: <http://purl.org/ontology/co/core#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX ontolex: <http://www.w3.org/ns/lemon/ontolex#>
PREFIX lila: <http://lila-erc.eu/ontologies/lila/>
PREFIX powla: <http://purl.org/powla/powla.owl#>
PREFIX corpora: <https://lila-erc.eu/ontologies/lila_corpora/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT ?label
WHERE {
  BIND(<http://lila-erc.eu/data/corpora/Lasla/id/corpus/SenecaAd%20Lucilium%20Epistulae%20Morales> AS ?documentURI)
    ?sentenceLayer powla:hasDocument ?documentURI.
      ?word powla:hasLayer ?sentenceLayer.
        ?word rdfs:label ?label
        }
        LIMIT 50
"""

response = requests.post(
    "https://lila-erc.eu/sparql/lila_knowledge_base/sparql",
    headers={
        "Accept": "application/sparql-results+json",
        "Content-Type": "application/x-www-form-urlencoded"
    },
    data={"query": query}
)

if response.status_code == 200:
    print(response.text)
    print(response.headers.get("Content-Type"))
    results = response.json()
    for result in results["results"]["bindings"]:
        print(result["label"]["value"])
else:
    print(f"Query failed with status code {response.status_code}")
