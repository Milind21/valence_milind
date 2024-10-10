from pymilvus import MilvusClient, model
import json

# query = "USER: Yes, all good. SYSTEM: Reservation is now confirmed!"
# num_demo = 3
def extract_demo(query,num_demo):
    client = MilvusClient("milvus_demo.db")

    embedding_fn = model.DefaultEmbeddingFunction()
    query_vectors = embedding_fn.encode_queries([query])

    res = client.search(
        collection_name="demo_collection",  # target collection
        data=query_vectors,  # query vectors
        limit=num_demo,  # number of returned entities
        output_fields=["text","score"],  # specifies fields to be returned
    )
    texts = []
    scores = []

# Extract text and score from the Milvus output
    for result in res[0]:  # Assuming you are interested in the first list of results
        texts.append(result['entity']['text'])
        scores.append(result['entity']['score'])
    return texts,scores
    
# extract_demo(query=query,num_demo=num_demo)