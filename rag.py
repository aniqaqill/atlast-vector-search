from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_voyageai import VoyageAIEmbeddings
import os

dbName = "book_mongodb_chunks"
collectionName = "chunked_data"
index = "vector_index"

vectorStore = MongoDBAtlasVectorSearch.from_connection_string(
    os.getenv("MONGODB_URI"),
    dbName + "." + collectionName,
    VoyageAIEmbeddings(voyage_api_key=os.getenv("VOYAGE_API_KEY"), model="voyage-3.5-lite"),
    index_name=index,
)

def query_data(query):
    retriever = vectorStore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 3,
            "pre_filter": { "hasCode": { "$eq": False } },
            "score_threshold": 0.01
        },
    )

    results = retriever.invoke(query)
    print(results)

    

query_data("When did MongoDB begin supporting multi-document transactions?")