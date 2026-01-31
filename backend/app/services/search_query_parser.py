"""
Advanced Search Query Parser

Implements query operators for filtering memories:
- tag:important - Filter by tag
- before:2024-01-01, after:2024-01-01, within:7d - Date filtering
- relates_to:memory_id - Filter by relationships
- scope:team, scope:personal - Filter by scope
- faceted: - Get faceted breakdown (tags, dates, authors)
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import re


class OperatorType(str, Enum):
    """Supported query operators."""
    TAG = "tag"
    BEFORE = "before"
    AFTER = "after"
    WITHIN = "within"
    RELATES_TO = "relates_to"
    SCOPE = "scope"
    FACETED = "faceted"
    AUTHOR = "author"
    STATUS = "status"


@dataclass
class QueryOperator:
    """Parsed query operator."""
    type: OperatorType
    value: str


@dataclass
class ParsedQuery:
    """Parsed search query with operators."""
    text: str  # Main text query
    operators: List[QueryOperator]
    faceted: bool = False
    limit: int = 50
    offset: int = 0


class SearchQueryParser:
    """Parse and validate search queries with operators."""
    
    # Regex patterns for each operator
    PATTERNS = {
        OperatorType.TAG: r'tag:(["\']?)(\w+)\1',
        OperatorType.BEFORE: r'before:(\d{4}-\d{2}-\d{2})',
        OperatorType.AFTER: r'after:(\d{4}-\d{2}-\d{2})',
        OperatorType.WITHIN: r'within:(\d+)([d|w|m])',  # days, weeks, months
        OperatorType.RELATES_TO: r'relates_to:([a-f0-9-]+)',  # UUID
        OperatorType.SCOPE: r'scope:(team|personal|shared)',
        OperatorType.AUTHOR: r'author:(["\']?)(\w+)\1',
        OperatorType.STATUS: r'status:(active|archived|deleted)',
        OperatorType.FACETED: r'\bfaceted\b',
    }
    
    def parse(self, query: str) -> ParsedQuery:
        """
        Parse a search query with operators.
        
        Examples:
            "AI research tag:important before:2024-01-01"
            "performance issues scope:team within:30d"
            "relates_to:abc123 faceted"
        
        Args:
            query: Raw query string with optional operators
            
        Returns:
            ParsedQuery with text and operators
        """
        query = query.strip()
        operators: List[QueryOperator] = []
        text_parts = []
        faceted = False
        
        # Extract faceted keyword
        if re.search(self.PATTERNS[OperatorType.FACETED], query):
            faceted = True
            query = re.sub(self.PATTERNS[OperatorType.FACETED], '', query).strip()
        
        # Extract all operators
        for op_type, pattern in self.PATTERNS.items():
            if op_type == OperatorType.FACETED:
                continue  # Already handled
            
            for match in re.finditer(pattern, query):
                if op_type in (OperatorType.BEFORE, OperatorType.AFTER):
                    value = match.group(1)
                elif op_type == OperatorType.WITHIN:
                    value = match.group(1) + match.group(2)  # "30d", "2w"
                elif op_type == OperatorType.TAG:
                    value = match.group(2)  # Quoted or unquoted
                elif op_type == OperatorType.AUTHOR:
                    value = match.group(2)
                else:
                    value = match.group(1)
                
                operators.append(QueryOperator(type=op_type, value=value))
                # Remove from query
                query = query.replace(match.group(0), '')
        
        # Remaining text is the main query
        text = ' '.join(query.split()).strip()
        
        return ParsedQuery(
            text=text,
            operators=operators,
            faceted=faceted,
            limit=50,
            offset=0
        )
    
    def validate(self, parsed: ParsedQuery) -> Tuple[bool, Optional[str]]:
        """
        Validate parsed query for errors.
        
        Args:
            parsed: ParsedQuery to validate
            
        Returns:
            (is_valid, error_message)
        """
        for op in parsed.operators:
            if op.type == OperatorType.BEFORE or op.type == OperatorType.AFTER:
                try:
                    datetime.strptime(op.value, '%Y-%m-%d')
                except ValueError:
                    return False, f"Invalid date format: {op.value}. Use YYYY-MM-DD"
            
            elif op.type == OperatorType.WITHIN:
                if not re.match(r'^\d+[dwm]$', op.value):
                    return False, f"Invalid duration: {op.value}. Use format like '30d', '2w', '1m'"
            
            elif op.type == OperatorType.SCOPE:
                if op.value not in ('team', 'personal', 'shared'):
                    return False, f"Invalid scope: {op.value}. Use team, personal, or shared"
            
            elif op.type == OperatorType.STATUS:
                if op.value not in ('active', 'archived', 'deleted'):
                    return False, f"Invalid status: {op.value}. Use active, archived, or deleted"
        
        return True, None
    
    def to_filters(self, parsed: ParsedQuery) -> Dict[str, Any]:
        """
        Convert parsed operators to database filters.
        
        Args:
            parsed: ParsedQuery
            
        Returns:
            Dict of filters for MemoryService
        """
        filters = {}
        
        for op in parsed.operators:
            if op.type == OperatorType.TAG:
                if 'tags' not in filters:
                    filters['tags'] = []
                filters['tags'].append(op.value)
            
            elif op.type == OperatorType.BEFORE:
                filters['before_date'] = datetime.strptime(op.value, '%Y-%m-%d')
            
            elif op.type == OperatorType.AFTER:
                filters['after_date'] = datetime.strptime(op.value, '%Y-%m-%d')
            
            elif op.type == OperatorType.WITHIN:
                # Parse duration
                match = re.match(r'(\d+)([dwm])', op.value)
                amount, unit = int(match.group(1)), match.group(2)
                
                if unit == 'd':
                    delta = timedelta(days=amount)
                elif unit == 'w':
                    delta = timedelta(weeks=amount)
                else:  # 'm'
                    delta = timedelta(days=amount * 30)
                
                filters['after_date'] = datetime.utcnow() - delta
            
            elif op.type == OperatorType.RELATES_TO:
                filters['related_to'] = op.value
            
            elif op.type == OperatorType.SCOPE:
                filters['scope'] = op.value
            
            elif op.type == OperatorType.AUTHOR:
                filters['created_by'] = op.value
            
            elif op.type == OperatorType.STATUS:
                filters['status'] = op.value
        
        return filters


# Global parser instance
_parser = SearchQueryParser()


def parse_search_query(query: str) -> ParsedQuery:
    """Parse a search query with operators."""
    return _parser.parse(query)


def validate_query(parsed: ParsedQuery) -> Tuple[bool, Optional[str]]:
    """Validate a parsed query."""
    return _parser.validate(parsed)


def query_to_filters(parsed: ParsedQuery) -> Dict[str, Any]:
    """Convert parsed query to database filters."""
    return _parser.to_filters(parsed)
