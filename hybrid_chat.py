# hybrid_chat.py
import json
import re
from typing import List
from google import genai
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from neo4j import GraphDatabase
import config
load_dotenv()

TOP_K = 10

INDEX_NAME = config.PINECONE_INDEX_NAME

client = genai.Client(api_key=config.GOOGLE_API_KEY)
pc = Pinecone(api_key=config.PINECONE_API_KEY)

if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating managed index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=config.PINECONE_VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east1-gcp")
    )

index = pc.Index(INDEX_NAME)

driver = GraphDatabase.driver(
    config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
)

from functools import lru_cache
@lru_cache(maxsize=1000)
def _cached_embed(text: str) -> tuple:
    resp = client.models.embed_content(
        model="gemini-embedding-001",
        contents=[text]
    )
    return tuple(resp.embeddings[0].values)

def embed_text(text: str) -> List[float]:
    return list(_cached_embed(text))
    
    return _cached_embed(text)

def pinecone_query(query_text: str, top_k=TOP_K):
    vec = embed_text(query_text)
    res = index.query(
        vector=vec,
        top_k=top_k,
        include_metadata=True,
        include_values=False
    )
    return res["matches"]

@lru_cache(maxsize=1000)  
def _fetch_single_node_context(node_id: str) -> tuple:
    facts = []
    with driver.session() as session:
        q = (
            "MATCH (n:Entity {id:$nid})-[r]-(m:Entity) "
            "RETURN type(r) AS rel, labels(m) AS labels, m.id AS id, "
            "m.name AS name, m.type AS type, m.description AS description "
            "LIMIT 10"
        )
        recs = session.run(q, nid=node_id)
        for r in recs:
            facts.append((
                node_id,
                r["rel"],
                r["id"],
                r["name"],
                r["description"] or "",
                tuple(r["labels"]) 
            ))
    return tuple(facts)  

def fetch_graph_context(node_ids: List[str], neighborhood_depth=1):
    facts = []
    for nid in node_ids:
        cached_facts = _fetch_single_node_context(nid)
        for f in cached_facts:
            facts.append({
                "source": f[0],
                "rel": f[1],
                "target_id": f[2],
                "target_name": f[3],
                "target_desc": f[4],
                "labels": list(f[5])
            })
    return facts
    print("DEBUG: Graph facts:")
    print(len(facts))
    return facts

def build_prompt(user_query, pinecone_matches, graph_facts):
    system = (
        "You are a helpful travel assistant."
        " First, carefully read the user's query."
        " Then, review the semantic search results and graph facts provided."
        " Think step-by-step:"
        " 1. Identify the main intent and entities in the user's question."
        " 2. Examine the top semantic matches and their metadata."
        " 3. Analyze the graph facts to find relevant relationships and context."
        " 4. Reason about the best answer, combining semantic and graph information."
        " 5. Present your answer clearly, with 2–3 actionable tips or itinerary steps."
        " 6. If helpful, cite node ids for specific places or attractions."
        " Use chain-of-thought reasoning and explain your steps briefly before the final answer."
        " Avoid internal system references; focus on a natural, user-friendly explanation."
    )

    vec_context = []
    for m in pinecone_matches[:TOP_K]:
        meta = m["metadata"]
        score = m.get("score", None)
        relevance = "Very Relevant" if score > 0.8 else "Relevant" if score > 0.6 else "Somewhat Relevant"
        location_info = f"in {meta.get('city')}" if meta.get('city') else ""
        
        snippet = (
            f" {meta.get('name', '')} ({relevance}): "
            f"A {meta.get('type', 'place')} {location_info}. "
        ).strip()
        
        if meta.get('description'):
            snippet += f"\n  Description: {meta.get('description')}..."
            
        vec_context.append(snippet)

    grouped_facts = {}
    for f in graph_facts:
        if f['source'] not in grouped_facts:
            grouped_facts[f['source']] = []
        grouped_facts[f['source']].append(f)
    
    graph_context = []
    for source, facts in grouped_facts.items():
        relationships = []
        for f in facts:
            rel_type = f['rel'].lower().replace('_', ' ')
            desc = f"which {f['target_desc']}..." if f['target_desc'] else ""
            relationships.append(f"- {rel_type} {f['target_name']} {desc}")
        
        if relationships:
            graph_context.extend(relationships)
    #print("*****************************************")
    #print(vec_context, graph_context)
    #print("*****************************************")

    # Use TOP_K to keep prompt context sizes consistent with the number of
    # results retrieved from the vector DB. For graph context we include up to
    # twice as many relationship lines to give more connective information.
    vec_limit = TOP_K
    graph_limit = TOP_K * 2

    prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content": (
            f"Question: {user_query}\n\n"
            "Context from vector database:\n" + "\n".join(vec_context[:vec_limit]) + "\n\n"
            "Context from graph database:\n" + "\n".join(graph_context[:graph_limit]) + "\n\n"
            "Please help the user by:"
            "\n1. First, explain which places and information are most relevant to their question"
            "\n2. Then, explain how these places are connected and why they're interesting"
            "\n3. Finally, provide a clear, practical answer with specific recommendations"
            "\n\nMake your response natural and friendly, as if you're a knowledgeable local guide."
        )}
    ]
    return prompt

def search_summary(matches: List[dict], graph_facts: List[dict]) -> dict:
    
    cities = {}
    for match in matches[:5]:
        meta = match["metadata"]
        city = meta.get("city", "Other")
        if city not in cities:
            cities[city] = []
        cities[city].append({
            "name": meta.get("name", "Unknown"),
            "type": meta.get("type", "place"),
            "score": match.get("score", 0),
            "id": match["id"]
        })
    
    # Find key relationships
    relationships = {}
    for fact in graph_facts:
        source = fact["source"]
        if source not in relationships:
            relationships[source] = []
        relationships[source].append({
            "type": fact["rel"],
            "target": fact["target_name"],
            "description": fact["target_desc"][:100] + "..." if len(fact["target_desc"]) > 100 else fact["target_desc"]
        })
    
    return {
        "places_by_city": cities,
        "key_relationships": relationships
    }

def call_chat(prompt_messages):
    resp = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt_messages,
    )
    return resp.text

# -----------------------------
# Interactive chat
# -----------------------------
def interactive_chat():
    while True:
        query = input("\nEnter your travel question: ").strip()
        if not query or query.lower() in ("exit","quit"):
            break

        matches = pinecone_query(query, top_k=TOP_K)
        match_ids = [m["id"] for m in matches]
        graph_facts = fetch_graph_context(match_ids)
        prompt = build_prompt(query, matches, graph_facts)
        answer = call_chat(str(prompt))
        
        print("\n=== Assistant Answer ===\n")
        print(answer)
        print("\n=== End ===\n")

if __name__ == "__main__":
    #print(embed_text("Test embedding"))
    interactive_chat()
