import os
import pandas as pd
from dotenv import load_dotenv
from rag_utils import prepare_docs_from_rows, upsert_documents, get_chroma

load_dotenv()

def main():
    csv_path = "data/data.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Create it first.")

    df = pd.read_csv(csv_path)
    required = {"id","question","answer"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {required}")

    rows = df.to_dict(orient="records")
    docs = prepare_docs_from_rows(rows)
    upsert_documents(docs)

    # show count
    _, col = get_chroma()
    print("✅ Ingested. Total docs in collection:", col.count())

if __name__ == "__main__":
    main()
