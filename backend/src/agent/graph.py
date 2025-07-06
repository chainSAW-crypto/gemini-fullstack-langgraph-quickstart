import os
import json
import re

from agent.tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from tavily import TavilyClient

from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)

from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
)
from agent.utils import (
    get_research_topic,
)

load_dotenv()

if os.getenv("GROQ_API_KEY") is None:
    raise ValueError("GROQ_API_KEY is not set")

if os.getenv("TAVILY_API_KEY") is None:
    raise ValueError("TAVILY_API_KEY is not set")

# Valid model names for Groq API
VALID_GROQ_MODELS = [
    "deepseek-r1-distill-llama-70b",
    "llama-3.1-70b-versatile", 
    "mixtral-8x7b-32768",
    "llama-3.3-70b-versatile"
]

def get_validated_model(state_model: str, fallback_model: str) -> str:
    """
    Validates and returns a valid Groq model name.
    
    Args:
        state_model: Model name from state (may be invalid)
        fallback_model: Fallback model from configuration
    
    Returns:
        Valid model name
    """
    if state_model and state_model in VALID_GROQ_MODELS:
        print(f"Using model from state: {state_model}")
        return state_model
    
    if fallback_model in VALID_GROQ_MODELS:
        print(f"Using fallback model: {fallback_model} (invalid state model: {state_model})")
        return fallback_model
    
    # Final safety net
    default_model = "deepseek-r1-distill-llama-70b"
    print(f"Using default model: {default_model} (invalid state: {state_model}, invalid fallback: {fallback_model})")
    return default_model


# Nodes
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates search queries based on the User's question.

    Uses DeepSeek-R1 to create optimized search queries for web research based on
    the User's question with advanced reasoning capabilities.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated queries
    """
    configurable = Configuration.from_runnable_config(config)

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # Get validated model for query generation
    query_model = get_validated_model(
        state.get("reasoning_model"), 
        configurable.query_generator_model
    )
    
    # init model
    llm = ChatGroq(
        model=query_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    # Format the prompt with JSON output instruction
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    
    # Add JSON format instruction to the prompt
    json_prompt = f"""{formatted_prompt}

Please provide your response in JSON format with the following structure:
{{"query": ["search query 1", "search query 2", "search query 3"]}}

Make sure to provide exactly {state["initial_search_query_count"]} search queries."""

    # Generate the search queries
    try:
        response = llm.invoke(json_prompt)
        content = response.content
        
        # Try to parse JSON from the response
        import json
        import re
        
        # Extract JSON from the response (handle markdown code blocks)
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON without code blocks
            json_match = re.search(r'\{.*?"query".*?\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # Fallback: create queries from the text response
                lines = content.strip().split('\n')
                queries = [line.strip().strip('-').strip() for line in lines if line.strip() and not line.startswith('#')]
                queries = [q for q in queries if len(q) > 10][:state["initial_search_query_count"]]
                if not queries:
                    queries = [get_research_topic(state["messages"])]
                return {"search_query": queries}
        
        parsed_result = json.loads(json_str)
        queries = parsed_result.get("query", [])
        
        # Ensure we have the right number of queries
        if not queries:
            queries = [get_research_topic(state["messages"])]
        elif len(queries) > state["initial_search_query_count"]:
            queries = queries[:state["initial_search_query_count"]]
        
        return {"search_query": queries}
        
    except Exception as e:
        print(f"Error generating queries: {e}")
        # Fallback to the original research topic
        return {"search_query": [get_research_topic(state["messages"])]}


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs web research using Tavily Search API.

    Executes a web search using Tavily API and processes results with DeepSeek-R1.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """
    # Configure
    configurable = Configuration.from_runnable_config(config)
    
    # Initialize Tavily client
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    # Perform search with Tavily
    try:
        search_results = tavily_client.search(
            query=state["search_query"],
            search_depth="advanced",
            max_results=5,
            include_answer=True,
            include_raw_content=True
        )
    except Exception as e:
        print(f"Tavily search failed: {e}")
        # Return empty results if search fails
        return {
            "sources_gathered": [],
            "search_query": [state["search_query"]],
            "web_research_result": [f"Search failed for query: {state['search_query']}"],
        }
    
    # Get validated model for web research processing
    research_model = get_validated_model(
        state.get("reasoning_model"), 
        configurable.query_generator_model
    )
    
    # Initialize model for processing search results
    llm = ChatGroq(
        model=research_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    
    # Format search results for LLM processing
    search_context = "\n\n---\n\n".join([
        f"Title: {result['title']}\nURL: {result['url']}\nContent: {result['content'][:1000]}..."
        for result in search_results.get('results', [])
    ])
    
    # Get web searcher instructions and format prompt
    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
    )
    
    # Add search results to the prompt
    full_prompt = f"""{formatted_prompt}

Search Results:
{search_context}

Based on these search results, provide a comprehensive analysis and summary of the information related to: {state["search_query"]}
Include key insights, findings, and relevant details from the sources."""
    
    # Process with DeepSeek-R1
    try:
        response = llm.invoke(full_prompt)
        research_result = response.content
    except Exception as e:
        print(f"LLM processing failed: {e}")
        research_result = f"Failed to process search results for: {state['search_query']}"
    
    # Format sources for consistency with the original format
    sources_gathered = []
    for i, result in enumerate(search_results.get('results', [])):
        source = {
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "short_url": result.get("url", ""),  # Tavily URLs are already clean
            "value": result.get("url", "")
        }
        sources_gathered.append(source)
    
    return {
        "sources_gathered": sources_gathered,
        "search_query": [state["search_query"]],
        "web_research_result": [research_result],
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    configurable = Configuration.from_runnable_config(config)
    # Increment the research loop count and get the reasoning model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1

    # Get validated model for reflection
    reasoning_model = get_validated_model(
        state.get("reasoning_model"), 
        configurable.reflection_model
    )

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    
    # Add JSON format instruction to the prompt
    json_prompt = f"""{formatted_prompt}

Please provide your response in JSON format with the following structure:
{{
    "is_sufficient": true/false,
    "knowledge_gap": "description of knowledge gaps",
    "follow_up_queries": ["query 1", "query 2", ...]
}}

Make sure to return valid JSON."""

    # init Reasoning Model with thinking capabilities
    llm = ChatGroq(
        model=reasoning_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    
    try:
        response = llm.invoke(json_prompt)
        content = response.content
        
        # Try to parse JSON from the response
        import json
        import re
        
        # Extract JSON from the response (handle markdown code blocks)
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON without code blocks
            json_match = re.search(r'\{.*?"is_sufficient".*?\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # Fallback: assume research is sufficient
                return {
                    "is_sufficient": True,
                    "knowledge_gap": "Unable to parse reflection response",
                    "follow_up_queries": [],
                    "research_loop_count": state["research_loop_count"],
                    "number_of_ran_queries": len(state["search_query"]),
                }
        
        parsed_result = json.loads(json_str)
        
        return {
            "is_sufficient": parsed_result.get("is_sufficient", True),
            "knowledge_gap": parsed_result.get("knowledge_gap", ""),
            "follow_up_queries": parsed_result.get("follow_up_queries", []),
            "research_loop_count": state["research_loop_count"],
            "number_of_ran_queries": len(state["search_query"]),
        }
        
    except Exception as e:
        print(f"Error in reflection: {e}")
        # Fallback: assume research is sufficient
        return {
            "is_sufficient": True,
            "knowledge_gap": f"Error processing reflection: {e}",
            "follow_up_queries": [],
            "research_loop_count": state["research_loop_count"],
            "number_of_ran_queries": len(state["search_query"]),
        }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Get validated model for final answer generation
    reasoning_model = get_validated_model(
        state.get("reasoning_model"), 
        configurable.answer_model
    )

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    # init Reasoning Model with Groq
    llm = ChatGroq(
        model=reasoning_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    result = llm.invoke(formatted_prompt)

    # Format sources for final answer (simplified since Tavily URLs are clean)
    unique_sources = []
    seen_urls = set()
    for source in state["sources_gathered"]:
        if source["url"] not in seen_urls:
            unique_sources.append(source)
            seen_urls.add(source["url"])

    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": unique_sources,
    }


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Set the entrypoint as `generate_query`
# This means that this node is the first one called
builder.add_edge(START, "generate_query")
# Add conditional edge to continue with search queries in a parallel branch
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
# Reflect on the web research
builder.add_edge("web_research", "reflection")
# Evaluate the research
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
# Finalize the answer
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent")
