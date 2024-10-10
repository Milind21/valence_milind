from pymilvus import Milvus, CollectionSchema, FieldSchema, DataType
from sentence_transformers import SentenceTransformer
import json
import uuid  # To generate unique IDs

def connect_to_milvus(host='localhost', port='19530'):
    """Connect to the Milvus server."""
    try:
        milvus = Milvus(host=host, port=port)
        print("Connected to Milvus server.")
        return milvus
    except Exception as e:
        print(f"Error connecting to Milvus: {e}")
        return None

def create_collection(milvus, collection_name):
    """Create a collection in Milvus."""
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=36, is_primary=True),  # Primary Key
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500, is_primary=False),
        FieldSchema(name="emo", dtype=DataType.INT32, is_primary=False),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768, is_primary=False)  # Dimension depends on model used
    ]
    schema = CollectionSchema(fields, description="Text and Emo Collection")

    try:
        milvus.create_collection(collection_name, schema)
        print(f"Collection '{collection_name}' created.")
    except Exception as e:
        print(f"Error creating collection: {e}")

def insert_data(milvus, collection_name, data, model):
    """Insert data into the Milvus collection."""
    try:
        texts = [entry['text'] for entry in data]
        emos = [entry['emo'] for entry in data]
        
        # Generate embeddings using Sentence Transformers
        vectors = model.encode(texts).tolist()  # Convert to list for Milvus compatibility

        # Generate unique IDs for primary key
        ids = [str(uuid.uuid4()) for _ in range(len(data))]  # Generate a unique ID for each record

        # Prepare the data to insert
        records = [
            ids,      # IDs as primary keys
            texts,    # Texts
            emos,     # Emo values
            vectors   # Vectors
        ]
        milvus.insert(collection_name, records)
        print(f"Inserted {len(data)} records into '{collection_name}'.")
    except Exception as e:
        print(f"Error inserting data: {e}")
        
def check_collection_exists(milvus, collection_name):
    """Check if a collection exists in Milvus."""
    exists = milvus.has_collection(collection_name)
    if exists:
        print(f"Collection '{collection_name}' exists.")
    else:
        print(f"Collection '{collection_name}' does not exist.")
        
def get_collection_info(milvus, collection_name):
    """Get information about the collection."""
    try:
        info = milvus.describe_collection(collection_name)
        print(f"Collection '{collection_name}' info:")
        print(info)
    except Exception as e:
        print(f"Error retrieving collection info: {e}")

def count_records(milvus, collection_name):
    """Count the number of records in the collection."""
    try:
        count = milvus.count_entities(collection_name)
        print(f"Number of records in '{collection_name}': {count}")
    except Exception as e:
        print(f"Error counting records: {e}")


def main():
    ip_path = 'unbiased_data.json'  # Change to your dataset path
    
    # Load your dataset
    with open(ip_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    data = data[:1000]
    # Load Sentence Transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose a different model based on your needs

    # Connect to Milvus
    milvus = connect_to_milvus()

    if milvus is None:
        return  # Exit if connection failed

    collection_name = 'demo_collection'
    
    # Create the collection
    create_collection(milvus, collection_name)

    # Insert data into the collection
    insert_data(milvus, collection_name, data, model)
    
    check_collection_exists(milvus, collection_name)

    # Get information about the collection
    get_collection_info(milvus, collection_name)

    # Count records in the collection
    count_records(milvus, collection_name)

if __name__ == "__main__":
    main()
