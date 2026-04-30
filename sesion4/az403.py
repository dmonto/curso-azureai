import os
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient

client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name=os.getenv("AZURE_SEARCH_INDEX"),
    credential=DefaultAzureCredential(exclude_interactive_browser_credential=False),
)

print("Documentos en índice:", client.get_document_count())

results = client.search(
    search_text="workflow",
    top=5,
    select=["title", "source", "domain", "content"],
)

for item in results:
    print(item)