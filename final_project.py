"""
UdaPlay - AI Game Research Agent
Final Project Implementation

This agent answers questions about video games using:
1. RAG (Retrieval-Augmented Generation) over local game database
2. Web search (Tavily API) as fallback when local knowledge is insufficient
3. Evaluation system to assess retrieval quality
4. Structured reporting with source citations
"""

import os
import json
import sys
import importlib.util
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tavily import TavilyClient
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Only needed for Udacity workspace - check if 'pysqlite3' is available
if importlib.util.find_spec("pysqlite3") is not None:
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Import library components
from lib.vector_db import VectorStore
from lib.agents import Agent
from lib.tooling import tool
from lib.llm import LLM
from lib.parsers import PydanticOutputParser
from lib.documents import Document, Corpus
from openai import OpenAI

# Load environment variables
load_dotenv()

# Get API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_OPENAI_API_KEY = os.getenv("CHROMA_OPENAI_API_KEY") or OPENAI_API_KEY
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize ChromaDB with persistent storage
CHROMA_DB_PATH = os.path.join(SCRIPT_DIR, "chromadb")
chroma_client = chromadb.Client(Settings(persist_directory=CHROMA_DB_PATH))

# Create embedding function
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=CHROMA_OPENAI_API_KEY
)

# Get or create the vector store for games
GAMES_COLLECTION_NAME = "udaplay"
games_store: Optional[VectorStore] = None


# ============================================================================
# Part 1: Vector Database Setup
# ============================================================================

def setup_vector_database():
    """Initialize ChromaDB and load game data from JSON files."""
    global games_store
    
    # Try to get existing collection and check if it has data
    collection = None
    collection_has_data = False
    
    try:
        existing_collection = chroma_client.get_collection(name=GAMES_COLLECTION_NAME)
        temp_store = VectorStore(existing_collection)
        
        # Check if collection has data using _query with a test embedding
        try:
            # Get collection ID
            collection_id = existing_collection.id
            
            # Create a simple test embedding (1536 dimensions for ada-002)
            # We'll use a zero vector for testing - not ideal but works
            test_embedding = [[0.0] * 1536]  # ada-002 has 1536 dimensions
            
            # Try to query to see if collection has data
            test_results = chroma_client._query(
                collection_id=collection_id,
                query_embeddings=test_embedding,
                n_results=1,
                include=['documents', 'metadatas']
            )
            
            # Check if we got any results
            documents = test_results.get('documents', [[]])
            if documents and len(documents) > 0 and len(documents[0]) > 0:
                # Collection has data
                all_data = temp_store.get()
                total_count = len(all_data.get("ids", [])) if all_data.get("ids") else 0
                if total_count > 0:
                    print(f"Vector database '{GAMES_COLLECTION_NAME}' already contains {total_count} documents.")
                    games_store = temp_store
                    return games_store
                else:
                    collection_has_data = False
            else:
                collection_has_data = False
                
            # If no data, we'll use the existing collection and load data
            collection = existing_collection
            games_store = temp_store
            
        except Exception as e:
            # Collection exists but has issues, delete and recreate
            print(f"Existing collection has issues, recreating... ({e})")
            try:
                chroma_client.delete_collection(name=GAMES_COLLECTION_NAME)
            except Exception:
                pass
            collection = None
            collection_has_data = False
    except Exception:
        # Collection doesn't exist
        collection = None
        collection_has_data = False
    
    # Create a new collection if needed
    if collection is None:
        # Create collection without embedding function to avoid ChromaDB 0.3.23 bug
        # We'll compute embeddings manually and pass them directly
        collection = chroma_client.create_collection(
            name=GAMES_COLLECTION_NAME
            # Don't pass embedding_function - we'll handle embeddings manually
        )
        games_store = VectorStore(collection)
        print(f"Created new collection '{GAMES_COLLECTION_NAME}' (embeddings will be computed manually).")
        collection_has_data = False
    
    # Only load data if collection is empty
    if not collection_has_data:
        # Load game data from JSON files
        games_dir = os.path.join(SCRIPT_DIR, "games")
        if not os.path.exists(games_dir):
            raise FileNotFoundError(f"Games directory '{games_dir}' not found")
        
        documents = []
        for file_name in sorted(os.listdir(games_dir)):
            if not file_name.endswith(".json"):
                continue
            
            file_path = os.path.join(games_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                game = json.load(f)
            
            # Format content as specified: [Platform] Name (Year) - Description
            content = f"[{game['Platform']}] {game['Name']} ({game['YearOfRelease']}) - {game['Description']}"
            
            # Use file name (without extension) as ID
            doc_id = os.path.splitext(file_name)[0]
            
            # Create Document with metadata
            doc = Document(
                id=doc_id,
                content=content,
                metadata=game  # Store full game metadata
            )
            documents.append(doc)
        
        # Add all documents to the vector store
        if documents:
            # Workaround for ChromaDB 0.3.23 bug: manually compute embeddings
            # and pass them directly instead of relying on collection's embedding function
            contents = [doc.content for doc in documents]
            ids = [doc.id for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # Compute embeddings manually using OpenAI client directly
            # This works around both the ChromaDB bug and OpenAI API version issue
            # Use the same base_url as the LLM class for consistency
            openai_client = OpenAI(
                base_url="https://openai.vocareum.com/v1",
                api_key=CHROMA_OPENAI_API_KEY
            )
            embeddings_response = openai_client.embeddings.create(
                model="text-embedding-ada-002",  # or "text-embedding-3-small" if available
                input=contents
            )
            embeddings = [item.embedding for item in embeddings_response.data]
            
            # Add directly using chroma_client's _add method to bypass collection._client bug
            # The collection.id is a UUID that _add needs
            chroma_client._add(
                collection_id=collection.id,
                documents=contents,
                ids=ids,
                metadatas=metadatas,
                embeddings=embeddings
            )
            print(f"Loaded {len(documents)} games into vector database '{GAMES_COLLECTION_NAME}'")
    
    return games_store


# ============================================================================
# Part 2: Evaluation Model
# ============================================================================

class EvaluationReport(BaseModel):
    """Structured evaluation report for retrieval quality assessment."""
    useful: bool = Field(description="Whether the retrieved documents are useful to answer the question")
    description: str = Field(description="Detailed explanation about the evaluation result")


# ============================================================================
# Part 3: Agent Tools Implementation
# ============================================================================

@tool
def retrieve_game(query: str) -> List[Dict[str, Any]]:
    """
    Semantic search: Finds most results in the vector DB.
    
    Searches the vector database for games matching the query.
    
    Args:
        query: A question about game industry.
    
    Returns:
        List of game documents. Each element contains:
        - Platform: like Game Boy, Playstation 5, Xbox 360...
        - Name: Name of the Game
        - YearOfRelease: Year when that game was released for that platform
        - Description: Additional details about the game
    """
    global games_store, chroma_client
    
    try:
        if games_store is None:
            games_store = setup_vector_database()
        
        # Verify collection has data
        if games_store is None:
            return []
        
        # Compute query embedding manually (workaround for ChromaDB 0.3.23 bug)
        openai_client = OpenAI(
            base_url="https://openai.vocareum.com/v1",
            api_key=CHROMA_OPENAI_API_KEY
        )
        query_embedding_response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=[query]
        )
        query_embeddings = [item.embedding for item in query_embedding_response.data]
        
        # Query using chroma_client._query directly to bypass collection._client bug
        # Get the collection ID
        collection_id = games_store._collection.id
        
        # Use chroma_client._query method directly
        results = chroma_client._query(
            collection_id=collection_id,
            query_embeddings=query_embeddings,  # List of embedding vectors
            n_results=10,  # Get top 10 most similar results for better coverage
            include=['documents', 'distances', 'metadatas']
        )
        
        # Format results with similarity scores
        retrieved_games = []
        documents = results.get('documents', [[]])
        metadatas = results.get('metadatas', [[]])
        distances = results.get('distances', [[]])
        
        # Handle both single query and list of queries
        if documents and len(documents) > 0:
            doc_list = documents[0] if isinstance(documents[0], list) else documents
            meta_list = metadatas[0] if metadatas and len(metadatas) > 0 and isinstance(metadatas[0], list) else (metadatas if metadatas else [])
            dist_list = distances[0] if distances and len(distances) > 0 and isinstance(distances[0], list) else (distances if distances else [])
            
            for i, doc in enumerate(doc_list):
                metadata = meta_list[i] if i < len(meta_list) else {}
                # Convert distance to similarity score (1 - distance, since lower distance = higher similarity)
                distance = dist_list[i] if i < len(dist_list) else 1.0
                similarity = max(0.0, 1.0 - distance)  # Ensure non-negative
                
                game_info = {
                    "Platform": metadata.get("Platform", ""),
                    "Name": metadata.get("Name", ""),
                    "YearOfRelease": metadata.get("YearOfRelease", ""),
                    "Description": metadata.get("Description", ""),
                    "Genre": metadata.get("Genre", ""),
                    "Publisher": metadata.get("Publisher", ""),
                    "Content": doc,  # Include the full document content
                    "similarity": round(similarity, 4)  # Add similarity score for reasoning
                }
                retrieved_games.append(game_info)
        
        return retrieved_games if retrieved_games else []
        
    except Exception as e:
        # Return error information for debugging
        error_msg = f"Error in retrieve_game: {str(e)}"
        print(f"Error in retrieve_game: {error_msg}")
        # Return empty list instead of failing completely
        return []


@tool
def evaluate_retrieval(question: str, retrieved_docs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Evaluates whether retrieved documents are sufficient to answer a question.
    
    WORKFLOW: 
    1. First call retrieve_game(question) to search the database
    2. Then call this function with: evaluate_retrieval(question, retrieved_docs) where retrieved_docs is the result from retrieve_game
    
    Args:
        question: The original user question
        retrieved_docs: (Optional) List of game documents from retrieve_game(). 
                       If omitted or empty, returns useful=False.
    
    Returns:
        Dictionary with:
        - useful: bool - whether documents can answer the question
        - description: str - explanation of the evaluation
    """
    # Handle missing or empty retrieved_docs
    if retrieved_docs is None or retrieved_docs == [] or (isinstance(retrieved_docs, list) and len(retrieved_docs) == 0):
        return {
            "useful": False,
            "description": "No retrieved documents provided. Please call retrieve_game() first to get documents, then pass the result as retrieved_docs parameter.",
            "confidence": 0.0,
            "num_documents": 0
        }
    # Create LLM instance for evaluation
    llm_judge = LLM(
        model="gpt-4o-mini",
        temperature=0.0,
        api_key=OPENAI_API_KEY
    )
    
    # Format retrieved documents for the prompt
    docs_text = "\n\n".join([
        f"Game: {doc.get('Name', 'Unknown')} ({doc.get('YearOfRelease', 'Unknown')})\n"
        f"Platform: {doc.get('Platform', 'Unknown')}\n"
        f"Description: {doc.get('Description', 'No description')}"
        for doc in retrieved_docs
    ])
    
    # Create evaluation prompt
    evaluation_prompt = f"""Your task is to evaluate if the documents are enough to respond to the query. 
Give a detailed explanation, so it's possible to take an action to accept it or not.

Question: {question}

Retrieved Documents:
{docs_text}

Evaluate whether these documents contain sufficient information to answer the question.
Consider:
1. Do the documents directly address the question?
2. Is there enough detail to provide a complete answer?
3. Are there any gaps in information that would require additional sources?

Provide your evaluation with a clear explanation."""

    # Get structured evaluation from LLM
    try:
        response = llm_judge.invoke(
            input=evaluation_prompt,
            response_format=EvaluationReport
        )
        
        # Parse the structured response
        parser = PydanticOutputParser(model_class=EvaluationReport)
        evaluation = parser.parse(response)
        
        # Calculate confidence based on similarity scores if available
        confidence = 0.5  # Default confidence
        if retrieved_docs and len(retrieved_docs) > 0:
            # Get best similarity score from retrieved docs
            similarities = [doc.get("similarity", 0.0) for doc in retrieved_docs if "similarity" in doc]
            if similarities:
                best_similarity = max(similarities)
                avg_similarity = sum(similarities) / len(similarities)
                # Confidence is based on both best match and average quality
                confidence = (best_similarity * 0.7 + avg_similarity * 0.3)
        
        return {
            "useful": evaluation.useful,
            "description": evaluation.description,
            "confidence": round(confidence, 3),
            "num_documents": len(retrieved_docs) if retrieved_docs else 0
        }
    except Exception as e:
        # Fallback evaluation
        print(f"Warning: Evaluation parsing failed: {e}")
        # Simple heuristic: if we have documents, assume they might be useful
        # Calculate confidence based on similarity scores if available
        confidence = 0.3  # Low confidence for fallback
        if retrieved_docs and len(retrieved_docs) > 0:
            similarities = [doc.get("similarity", 0.0) for doc in retrieved_docs if "similarity" in doc]
            if similarities:
                best_similarity = max(similarities)
                confidence = best_similarity * 0.5  # Lower confidence for fallback
        
        return {
            "useful": len(retrieved_docs) > 0,
            "description": f"Evaluation parsing failed. Found {len(retrieved_docs)} documents. Error: {str(e)}",
            "confidence": round(confidence, 3),
            "num_documents": len(retrieved_docs) if retrieved_docs else 0
        }


@tool
def game_web_search(question: str) -> Dict[str, Any]:
    """
    Search the web for information about video games using Tavily API.
    
    This tool performs a REAL web search using the Tavily API to fetch live information
    from the internet when the internal database doesn't have sufficient information.
    
    Args:
        question: A question about game industry.
    
    Returns:
        Dictionary containing:
        - answer: Direct answer from Tavily (if available)
        - results: List of search results with title, url, content, score
        - search_metadata: Metadata about the search including query and result_count
    """
    # Validate API key
    if not TAVILY_API_KEY:
        raise ValueError(
            "TAVILY_API_KEY not found in environment variables. "
            "Please set TAVILY_API_KEY in your .env file or environment variables."
        )
    
    try:
        # Initialize Tavily client with API key
        client = TavilyClient(api_key=TAVILY_API_KEY)
        
        # Perform REAL web search using Tavily API
        # This makes an actual HTTP request to Tavily's servers
        search_result = client.search(
            query=question,
            search_depth="advanced",  # Use advanced search for better results
            include_answer=True,  # Include AI-generated answer
            include_raw_content=False,  # Don't include raw HTML content
            include_images=False  # Don't include images
        )
        
        # Extract and format the results from Tavily API response
        answer = search_result.get("answer", "")
        results = search_result.get("results", [])
        
        # Format each result for better readability
        formatted_results_list = []
        for result in results:
            formatted_results_list.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "score": result.get("score", 0.0)
            })
        
        # Return structured response
        formatted_results = {
            "answer": answer,  # Tavily's AI-generated answer
            "results": formatted_results_list,  # List of search results
            "search_metadata": {
                "query": question,
                "result_count": len(formatted_results_list),
                "source": "tavily_api"  # Indicate this came from real API
            }
        }
        
        return formatted_results
        
    except Exception as e:
        # Re-raise with more context for debugging
        error_msg = f"Tavily API search failed: {str(e)}"
        print(f"Error in game_web_search: {error_msg}")
        raise RuntimeError(
            f"Web search failed. Please check your TAVILY_API_KEY and network connection. "
            f"Error details: {str(e)}"
        )


# ============================================================================
# Part 4: Agent Implementation
# ============================================================================

def create_udaplay_agent() -> Agent:
    """
    Create and configure the UdaPlay agent with all tools.
    
    Returns:
        Configured Agent instance
    """
    # System instructions for the agent
    system_instructions = """You are UdaPlay, an AI Research Agent specialized in answering questions about video games.

Your capabilities:
1. Search internal knowledge base (vector database) for game information
2. Evaluate whether retrieved information is sufficient to answer questions
3. Search the web when internal knowledge is insufficient or when current information is needed
4. Generate clear, well-structured answers that cite your sources

CRITICAL WORKFLOW - You MUST follow these steps in order for EVERY question:

STEP 1: ALWAYS start by calling retrieve_game(query) with the user's question
  - This searches your internal database of games
  - It returns a list of game documents with information like Name, Platform, YearOfRelease, Description, Publisher
  
STEP 2: ALWAYS call evaluate_retrieval(question=user_question, retrieved_docs=result_from_step_1)
  - You MUST pass BOTH parameters: question (the original question) AND retrieved_docs (the list from retrieve_game)
  - This evaluates whether the retrieved documents can answer the question
  - It returns {"useful": True/False, "description": "..."}
  
STEP 3: Based on evaluation result:
  - If useful=True: Use the retrieved documents to answer the question. DO NOT search the web.
  - If useful=False: Then call game_web_search(question) to find additional information
  
STEP 4: Generate your final answer
  - If you used internal database: Cite "internal database" or "vector database"
  - If you used web search: Cite the web sources
  - Combine information from multiple sources if needed

IMPORTANT RULES:
- NEVER skip retrieve_game() - always search the internal database first
- NEVER call game_web_search() without first calling retrieve_game() and evaluate_retrieval()
- ALWAYS pass the retrieved_docs parameter to evaluate_retrieval() - it's required
- If retrieve_game() returns an empty list, then evaluate_retrieval() will return useful=False, and THEN you can use web search

EXAMPLE WORKFLOW:
  User: "When was Pokémon Gold released?"
  
  Step 1: retrieved = retrieve_game("When was Pokémon Gold released?")
           # Returns: [{"Name": "Pokémon Gold", "YearOfRelease": 1999, ...}, ...]
  
  Step 2: evaluation = evaluate_retrieval(
              question="When was Pokémon Gold released?",
              retrieved_docs=retrieved  # Pass the result from step 1
          )
           # Returns: {"useful": True, "description": "..."}
  
  Step 3: Since useful=True, use the retrieved documents to answer
          Answer: "Pokémon Gold was released in 1999 for the Game Boy Color."

Be thorough, accurate, and helpful. Always start with the internal database before searching the web."""

    # Create agent with all tools
    agent = Agent(
        model_name="gpt-4o-mini",
        instructions=system_instructions,
        tools=[retrieve_game, evaluate_retrieval, game_web_search],
        temperature=0.7
    )
    
    return agent


# ============================================================================
# Part 5: Main Execution and Testing
# ============================================================================

def main():
    """Main execution function to test the UdaPlay agent."""
    print("=" * 80)
    print("UdaPlay - AI Game Research Agent")
    print("=" * 80)
    print()
    
    # Setup vector database
    print("Setting up vector database...")
    setup_vector_database()
    print()
    
    # Create agent
    print("Creating UdaPlay agent...")
    agent = create_udaplay_agent()
    print("Agent created successfully!")
    print()
    
    # Test queries - including fallback test and follow-up
    test_queries = [
        ("When was Pokémon Gold and Silver released?", None),
        ("Which one was the first 3D platformer Mario game?", None),
        ("Was Mortal Kombat X released for PlayStation 5?", None),
        ("Who developed FIFA 21?", None),
        ("What is the publisher name of Grand Theft Auto: San Andreas?", None),
        ("What platform was Pokémon Red launched on?", None),
        # Fallback test: 2024/2025 game that won't be in database
        ("When was Palworld released and who developed it?", None),
        # Follow-up question to demonstrate conversation history
        ("What platform was it released on?", "default"),  # Uses same session
    ]
    
    print("=" * 80)
    print("Testing Agent with Sample Queries")
    print("=" * 80)
    print()
    
    for i, query_info in enumerate(test_queries, 1):
        if isinstance(query_info, tuple):
            query, session_id = query_info
        else:
            query = query_info
            session_id = None
        
        print(f"Query {i}: {query}")
        print("-" * 80)
        
        try:
            # Invoke agent with session_id for conversation history
            run = agent.invoke(query, session_id=session_id or "default")
            
            # Get final state and extract reasoning trace
            final_state = run.get_final_state()
            messages = final_state.get("messages", [])
            
            # Extract reasoning trace from tool calls
            reasoning_steps = []
            source_type = "internal_database"
            confidence = None
            num_docs = 0
            
            web_search_called = False
            last_evaluation_useful = True
            
            for msg in messages:
                # Track tool calls for reasoning
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tool_name = tool_call.function.name
                        reasoning_steps.append(f"→ Called {tool_name}()")
                        if tool_name == "game_web_search":
                            web_search_called = True
                
                # Extract tool results for reasoning
                if hasattr(msg, 'name') and msg.name:  # ToolMessage
                    tool_name = msg.name
                    try:
                        import json, ast
                        # Try to parse JSON content
                        content_str = msg.content if isinstance(msg.content, str) else str(msg.content)
                        try:
                            result = json.loads(content_str)
                        except (json.JSONDecodeError, TypeError):
                            try:
                                result = ast.literal_eval(content_str)
                            except (ValueError, SyntaxError):
                                result = content_str
                        
                        # Handle retrieve_game results (list of dicts)
                        if tool_name == "retrieve_game":
                            if isinstance(result, list):
                                num_docs = len(result)
                                if num_docs > 0:
                                    best_sim = result[0].get("similarity", None) if isinstance(result[0], dict) else None
                                    reasoning_steps.append(
                                        f"→ Retrieved {num_docs} documents from internal database"
                                    )
                                    if best_sim is not None:
                                        reasoning_steps.append(f"  Best similarity score: {best_sim:.3f}")
                                else:
                                    reasoning_steps.append("→ No documents found in internal database")
                        
                        # Handle evaluate_retrieval results (dict with useful, confidence, etc.)
                        elif tool_name == "evaluate_retrieval":
                            if isinstance(result, dict):
                                useful = result.get("useful", False)
                                last_evaluation_useful = useful
                                confidence_val = result.get("confidence", None)
                                num_docs_val = result.get("num_documents", result.get("num_docs", 0))
                                eval_desc = result.get("description", "")
                                
                                confidence = confidence_val  # always capture, even if 0.0
                                
                                conf_str = f"{confidence_val:.3f}" if confidence_val is not None else "N/A"
                                reasoning_steps.append(
                                    f"→ Evaluation: useful={useful} | confidence={conf_str} | documents={num_docs_val}"
                                )
                                if eval_desc:
                                    reasoning_steps.append(f"  Reasoning: {eval_desc[:150]}{'...' if len(eval_desc) > 150 else ''}")
                                
                                if not useful:
                                    reasoning_steps.append("→ Decision: Internal database insufficient, falling back to web search")
                        
                        # Handle game_web_search results
                        elif tool_name == "game_web_search":
                            if isinstance(result, dict):
                                source_type = "web_search"
                                reasoning_steps.append("→ Web search completed, found external information")
                                if "result_count" in result:
                                    reasoning_steps.append(f"  Found {result.get('result_count', 0)} web results")
                    except Exception as e:
                        # Silently continue if parsing fails
                        pass
            
            # Determine final source type based on tool calls
            if web_search_called:
                source_type = "web_search"
            elif not last_evaluation_useful:
                source_type = "web_search"
            else:
                source_type = "internal_database"
            
            # Find the last AI message with content (the final answer)
            final_answer = None
            for msg in reversed(messages):
                if hasattr(msg, 'content') and msg.content and not hasattr(msg, 'name'):
                    final_answer = msg.content
                    break
            
            # Display results with reasoning
            print("\n[REASONING TRACE]")
            if reasoning_steps:
                for step in reasoning_steps:
                    print(f"  {step}")
            else:
                print("  No tool calls detected in this query.")
            
            print(f"\n[RESULT]")
            if final_answer:
                print(f"Answer: {final_answer}")
            else:
                print("No answer generated.")
            
            print(f"\n[METADATA]")
            print(f"Source: {source_type}")
            conf_display = f"{confidence:.3f}" if confidence is not None else "N/A"
            print(f"Confidence: {conf_display}")
            if num_docs > 0:
                print(f"Documents retrieved: {num_docs}")
            
            # Show token usage if available
            total_tokens = final_state.get("total_tokens", 0)
            if total_tokens > 0:
                print(f"Token usage: {total_tokens} tokens")
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print()
        print("=" * 80)
        print()


if __name__ == "__main__":
    main()

