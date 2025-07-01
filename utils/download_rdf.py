import requests
from rdflib import Graph
import io

query = """
PREFIX dc: <http://purl.org/dc/elements/1.1/>
PREFIX powla: <http://purl.org/powla/powla.owl#>

CONSTRUCT {
  ?doc ?p ?o .
}
WHERE {
  ?doc ?pred powla:Document ;
       dc:title ?title .

  ?doc ?p ?o .
}
ORDER BY ?title
LIMIT 100
"""

endpoint_url = "https://lila-erc.eu/sparql/lila_knowledge_base/sparql?format=json"

headers = {
    "Accept": "application/n-triples,*/*;q=0.9",
    "Content-Type": "application/x-www-form-urlencoded",
    "User-Agent": "Mozilla/5.0"
}

response = requests.post(endpoint_url, data={"query": query}, headers=headers)

print(f"Status: {response.status_code}")
print("Content-Type:", response.headers.get("Content-Type", ""))
print("First 500 chars of response:", response.text[:500])

# Try parsing as JSON-serialized RDF if needed
try:
    g = Graph()
    g.parse(data=response.text, format="json-ld")  # Adjust this if not JSON-LD compatible
    g.serialize("lila_construct.ttl", format="turtle")
    print(f"Saved {len(g)} triples to lila_construct.ttl")
except Exception as e:
    print("Failed to parse response as RDF:", e)
