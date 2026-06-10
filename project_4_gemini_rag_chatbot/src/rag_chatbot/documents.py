from pypdf import PdfReader
from rag_chatbot.config import DOCUMENTS_DIR

def load_text_file(path):
    return path.read_text(encoding="utf-8")

def load_pdf(path):
    reader = PdfReader(path)
    documents = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            documents.append(
                {
                    "text": text,
                    "source": path.name,
                    "page": page_num + 1,
                }
            )
    return documents

def load_documents():
    documents = []

    for path in DOCUMENTS_DIR.iterdir():
        if path.suffix.lower() in [".txt",".md"]:
            documents.append(
                {
                "text": load_text_file(path),
                "source":path.name,
                "page":None,
                }
            )
        elif path.suffix.lower() == ".pdf":
            documents.extend(load_pdf(path))
    return documents


def main():
    documents = load_documents()

    print(f"Loaded {len(documents)} documents:")
    
    for document  in documents:
        print("\nSource:", document["source"])
        print("Page:", document["page"])
        print(document["text"][:300])

if __name__ == "__main__":
    main()