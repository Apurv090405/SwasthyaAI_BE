"""LanceDB vector database service."""

from typing import Dict, List, Any, Optional

import lancedb
import pandas as pd

from config import settings


class LanceDBService:
    """Service for LanceDB vector database operations."""
    
    def __init__(self, db_path: str = None):
        """
        Initialize LanceDB service.
        
        Args:
            db_path: Path to LanceDB database (defaults to settings.LANCEDB_PATH)
        """
        self.db_path = db_path or settings.LANCEDB_PATH
        self._db = None
    
    @property
    def db(self):
        """Lazy load database connection."""
        if self._db is None:
            self._db = lancedb.connect(self.db_path)
        return self._db
    
    def store_embeddings(
        self, 
        data: List[Dict[str, Any]], 
        table_name: str = None
    ) -> int:
        """
        Store embeddings and text in LanceDB.
        
        Args:
            data: List of dictionaries with embeddings and metadata
            table_name: Name of the table to store data in
            
        Returns:
            Number of records stored
        """
        table_name = table_name or settings.LANCEDB_TABLE_NAME
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(data)
        
        # Check if table exists
        try:
            table = self.db.open_table(table_name)
            # Append to existing table
            table.add(df)
        except Exception:
            # Create new table if it doesn't exist
            # LanceDB will automatically detect the vector column
            table = self.db.create_table(table_name, df, mode="overwrite")
        
        return len(data)
    
    def search(
        self, 
        query_embedding: List[float], 
        top_k: int = None,
        table_name: str = None
    ) -> List[Dict[str, Any]]:
        """
        Search LanceDB vector store for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            table_name: Name of the table to search
            
        Returns:
            List of search results with metadata
        """
        top_k = top_k or settings.DEFAULT_TOP_K
        table_name = table_name or settings.LANCEDB_TABLE_NAME
        
        try:
            table = self.db.open_table(table_name)
        except Exception as e:
            raise ValueError(f"LanceDB table '{table_name}' not found: {str(e)}")
        
        # Search using the vector column (LanceDB auto-detects "vector" column)
        results = table.search(query_embedding).limit(top_k).to_pandas()
        
        hits = []
        for _, row in results.iterrows():
            hits.append({
                "id": row.get("id", ""),
                "score": row.get("_distance", None),
                "text": row.get("text", ""),
                "source": row.get("source", "")
            })
        
        return hits

    def list_sources(self, table_name: str = None) -> List[str]:
        """Return sorted unique ``source`` values in the table (full scan; OK for modest corpora)."""
        table_name = table_name or settings.LANCEDB_TABLE_NAME
        try:
            table = self.db.open_table(table_name)
        except Exception:
            return []
        try:
            df = table.to_pandas()
        except Exception:
            return []
        if df is None or df.empty or "source" not in df.columns:
            return []
        vals = {str(s).strip() for s in df["source"].dropna().unique() if str(s).strip()}
        return sorted(vals)

    def search_by_text(
        self,
        query_text: str,
        top_k: int = 5,
        source: Optional[str] = None,
        table_name: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Embed ``query_text`` and vector-search LanceDB, optionally filtering by ``source``.
        """
        if not query_text or not str(query_text).strip():
            return []
        q = str(query_text).strip()
        vec = settings.embedding_model.encode(
            q,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).tolist()
        fetch_k = top_k * 20 if source else top_k
        hits = self.search(vec, top_k=max(fetch_k, top_k), table_name=table_name)
        if source:
            hits = [h for h in hits if (h.get("source") or "") == source]
        return hits[:top_k]

