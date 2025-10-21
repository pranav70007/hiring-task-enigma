Weighing by relevance:

Workflow:
1. pinecone_query() takes in the query and returns the top 'k' matches along with metadata, from the pinecone database, and saves them as 'matches'
2. fetch_graph_context() takes in the 'id' attribute of all the matches and returns all the nodes at the given depth of the node with the given id. 

1. embed_text() returns the embeddings of the input and stores them as vec
2. index.query() returns the top k 
index.query(
        vector=vec,
        top_k=top_k,
        include_metadata=True,
        include_values=False
    )
 1. Group places by city.
