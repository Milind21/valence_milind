from pymilvus import MilvusClient, model, CollectionSchema, FieldSchema, DataType, Milvus
import json

#loading data
with open("unbiased_data.json", "r", encoding="utf-8") as file:
  data = json.load(file)
data = data[:1000]

# establish connection 
client = MilvusClient("milvus_demo.db")
collection_name = "demo_collection"
# create collection
schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=False,
)
schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=36, is_primary=True)
schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=500, is_primary=False)
schema.add_field(field_name="score", datatype=DataType.INT32, is_primary=False)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=768, is_primary=False)  # Dimension depends on model used

if client.has_collection(collection_name=collection_name):
    client.drop_collection(collection_name=collection_name)
client.create_collection(
    collection_name=collection_name,
    dimension=768,  # The vectors we will use in this demo has 768 dimensions
)
print(f"Collection '{collection_name}' created.")

# define embedding function paraphrase-albert-small-v2

embedding_fn = model.DefaultEmbeddingFunction()
# text strings to search from
docs = [item["text"] for item in data]
scores = [int(item["emo"]) for item in data]
vectors = embedding_fn.encode_documents(docs)
print("Dim:", embedding_fn.dim, vectors[0].shape)  # Dim: 768 (768,)

#creating data to put in the collection
data = [
    {"id": i, "vector": vectors[i], "text": docs[i], "score": scores[i]}
    for i in range(len(vectors))
]

print("Data has", len(data), "entities, each with fields: ", data[0].keys())
print("Vector dim:", len(data[0]["vector"]))

res = client.insert(collection_name="demo_collection", data=data)

# print(res)
