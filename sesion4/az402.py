import base64
import os
from pathlib import Path
from typing import Iterable

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from openai import AzureOpenAI

from pypdf import PdfReader
from docx import Document


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Falta la variable de entorno: {name}")
    return value


def make_safe_id(value: str) -> str:
    encoded = base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii")
    return encoded.rstrip("=")


def read_txt_or_md(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)

    return "\n".join(pages)


def read_docx(path: Path) -> str:
    document = Document(str(path))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return read_txt_or_md(path)

    if suffix == ".pdf":
        return read_pdf(path)

    if suffix == ".docx":
        return read_docx(path)

    raise ValueError(f"Extensión no soportada: {path.suffix}")


def chunk_text(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[str]:
    clean_text = " ".join(text.split())

    if not clean_text:
        return []

    chunks = []
    start = 0

    while start < len(clean_text):
        end = start + chunk_size
        chunk = clean_text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def iter_files(folder: Path) -> Iterable[Path]:
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)

token_provider = get_bearer_token_provider(
    credential,
    "https://cognitiveservices.azure.com/.default",
)

openai_client = AzureOpenAI(
    azure_endpoint=require_env("AOAI_ENDPOINT_PRIMARY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    azure_ad_token_provider=token_provider,
)

search_client = SearchClient(
    endpoint=require_env("AZURE_SEARCH_ENDPOINT"),
    index_name=require_env("AZURE_SEARCH_INDEX"),
    credential=credential,
)


def embed_text(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        model=require_env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        input=text,
    )
    return response.data[0].embedding


def build_documents_from_folder(
    folder: Path,
    domain: str,
) -> list[dict]:
    documents = []

    for path in iter_files(folder):
        print(f"Leyendo: {path}")

        text = read_document(path)
        chunks = chunk_text(text)

        print(f"  chunks: {len(chunks)}")

        for i, chunk in enumerate(chunks):
            source = str(path).replace("\\", "/")
            doc_id = make_safe_id(f"{source}::{i}")

            documents.append(
                {
                    "id": doc_id,
                    "title": path.stem,
                    "content": chunk,
                    "source": source,
                    "domain": domain,
                    "contentVector": embed_text(chunk),
                }
            )

    return documents


def upload_in_batches(documents: list[dict], batch_size: int = 50) -> None:
    total = len(documents)

    for start in range(0, total, batch_size):
        batch = documents[start:start + batch_size]
        result = search_client.upload_documents(documents=batch)

        succeeded = sum(1 for item in result if item.succeeded)
        print(f"Subidos {succeeded}/{len(batch)} documentos")


if __name__ == "__main__":
    folder = Path(os.getenv("RAG_DOCS_FOLDER", "docs"))
    domain = os.getenv("RAG_DOMAIN", "general")

    if not folder.exists():
        raise RuntimeError(f"No existe la carpeta: {folder}")

    docs = build_documents_from_folder(folder, domain)

    if not docs:
        print("No se han encontrado documentos para indexar.")
    else:
        upload_in_batches(docs)
        print(f"Indexación completada. Chunks subidos: {len(docs)}")