import json
from pymilvus import MilvusClient
from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference

# Initialize embedding model
local_embed_model = TextEmbeddingsInference(
    model_name="BAAI/bge-large-en-v1.5",
    base_url="http://127.0.0.1:8081",  # Adjust this URL to your inference server
    timeout=60,
    embed_batch_size=10,
)

# Connect to Milvus
client = MilvusClient(uri="http://localhost:19530")

# Query
query = "What are different components?"
embeddings = local_embed_model.get_query_embedding(query)

# Inspect collection schema (uncomment for debugging)
# collection_schema = client.describe_collection("app_milvus_db")
# print(collection_schema)



# collection_schema = client.describe_collection("app_milvus_db")
# print(json.dumps(collection_schema, indent=4))

# Perform search
res = client.search(
    collection_name="app_milvus_db",  # Replace with your collection name
    data=[embeddings],
    limit=5,  # Number of results
    # search_params = {
    # "metric_type": "IP",
    # "params": {
    #     "radius": 0.8, # Radius of the search circle
    #     "range_filter": 1.0 # Range filter to filter out vectors that are not within the search circle
    #     }
    # },  # Search parameters
    output_fields=["_node_content"],  # Replace with a valid field name, e.g., 'text' or 'metadata'
)

# print(res)

# Pretty print results
# result = json.dumps(res, indent=4)
# print(result)

result=""
for search_result in res[0]:  # Assuming `res` is a list containing the first batch of results
    # print("\n\n")
    # print(f"Search Result: {search_result}")  # Print the full dictionary to debug the structure
    # print(f"ID: {search_result.get('id', 'N/A')}, Distance: {search_result.get('distance', 'N/A')}")
    val=search_result.get('entity', 'No content available')
    if val != 'No content available':
        val2 = val.get('_node_content', 'No content available')
        if val2 != 'No content available':
            # Check if val2 is a dictionary
            if isinstance(val2, dict):
                # If it's a dictionary, safely access the "text" field
                text_content = val2.get("text", "No text available")
            else:
                # If it's a string, assume it's the text content
                text_content = val2 if val2 else "No text available"

            # Print the content
            # print("Doc")
            # print(text_content)

            parsed_content = json.loads(text_content)

            text_content = parsed_content.get("text", "No text available")

            # print(text_content)

            result+=text_content+"\n\n"
        else:
            print("No node content found.")
    else:
        print("No entity found.")
    
    print(result)
