"""Search service for health news and public health information."""

import json
from typing import Optional, List, Dict, Any

try:
    from serpapi import GoogleSearch
except ImportError:
    try:
        from serpapi.google_search import GoogleSearch
    except ImportError:
        GoogleSearch = None

from config import settings


class SearchService:
    """Service for searching health news via SerpAPI."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize search service.
        
        Args:
            api_key: SerpAPI key (defaults to settings.SERPAPI_KEY)
        """
        self.api_key = api_key or settings.SERPAPI_KEY
        if not self.api_key:
            raise ValueError("SERPAPI_KEY must be set for search functionality")
        
        if GoogleSearch is None:
            raise ImportError(
                "SerpAPI package not properly installed. "
                "Install with: pip install google-search-results"
            )
    
    def search_health_news(
        self, 
        query: str, 
        location: str = "",
        limit: int = None
    ) -> List[Dict[str, Any]]:
        """
        Search for recent health news, outbreaks, or public health advisories.
        
        Args:
            query: Search query (can be in English or Hindi)
            location: Optional location for localized results
            limit: Maximum number of results to return
            
        Returns:
            List of news hits with title, snippet, link, and source
        """
        limit = limit or settings.SEARCH_RESULTS_LIMIT
        
        params = {
            "engine": "google",
            "q": query,
            "api_key": self.api_key,
            "num": limit,
            "hl": settings.SEARCH_LANGUAGE,
            "gl": settings.SEARCH_REGION,
        }
        
        if location:
            params["location"] = location
        
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
        except Exception as e:
            raise RuntimeError(f"SerpAPI search failed: {str(e)}")
        
        hits = []
        
        # Try news results first
        for result in results.get("news_results", [])[:limit]:
            hits.append({
                "title": result.get("title"),
                "snippet": result.get("snippet"),
                "link": result.get("link"),
                "source": result.get("source")
            })
        
        # Fallback to organic results if no news results
        if not hits:
            for result in results.get("organic_results", [])[:limit]:
                hits.append({
                    "title": result.get("title"),
                    "snippet": result.get("snippet"),
                    "link": result.get("link"),
                    "source": result.get("displayed_link")
                })
        
        return hits

