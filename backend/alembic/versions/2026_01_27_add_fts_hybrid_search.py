"""Add full-text search support for hybrid search

Revision ID: 2026_01_27_0005_add_fts_hybrid_search
Revises: (aafe51a38844, 2026_01_27_0004_fix_admin_settings_types)
Create Date: 2026-01-27

This migration adds PostgreSQL full-text search capabilities to the memory_metadata
table for hybrid search (BM25 + vector). It adds:

1. tsvector column for pre-computed search vectors
2. GIN index for fast full-text search
3. Trigger to automatically update tsvector on INSERT/UPDATE
4. Configuration for search weights (title > content)

Performance characteristics:
- GIN index: ~30-50% of original table size
- Search performance: O(log n) with GIN index vs O(n) without
- BM25-style ranking via ts_rank_cd with normalization
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic
revision = 'add_fts_hybrid_search'
down_revision = 'backup_models'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Add full-text search support to memory_metadata table.
    
    Creates:
    - search_vector column (tsvector) for pre-computed FTS vectors
    - GIN index on search_vector for fast searches
    - Trigger function to automatically update search_vector
    - Trigger to call the function on INSERT/UPDATE
    """
    
    # 1. Add tsvector column for full-text search
    # Using 'simple' dictionary for multilingual support and exact matching
    # (can be changed to 'english' for better stemming if needed)
    op.execute("""
        ALTER TABLE memory_metadata
        ADD COLUMN search_vector tsvector;
    """)
    
    # 2. Create GIN index for fast full-text search
    # GIN (Generalized Inverted Index) is optimal for tsvector columns
    op.execute("""
        CREATE INDEX idx_memory_metadata_search_vector
        ON memory_metadata
        USING GIN (search_vector);
    """)
    
    # 3. Create trigger function to maintain search_vector
    # This function automatically updates search_vector on INSERT/UPDATE
    # Weight configuration: A (content_preview) > D (tags)
    # Note: MemoryMetadata doesn't have a separate 'title' field
    op.execute("""
        CREATE OR REPLACE FUNCTION memory_metadata_search_vector_update()
        RETURNS trigger AS $$
        BEGIN
            -- Build weighted tsvector:
            -- 'A' weight (highest): content_preview (acts as both title and content)
            -- 'D' weight (low): tags
            -- Using 'simple' config for multilingual + exact term matching
            NEW.search_vector :=
                setweight(to_tsvector('simple', COALESCE(NEW.content_preview, '')), 'A') ||
                setweight(to_tsvector('simple', array_to_string(COALESCE(NEW.tags, ARRAY[]::text[]), ' ')), 'D');
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # 4. Create trigger to call the function
    op.execute("""
        CREATE TRIGGER memory_metadata_search_vector_trigger
        BEFORE INSERT OR UPDATE ON memory_metadata
        FOR EACH ROW
        EXECUTE FUNCTION memory_metadata_search_vector_update();
    """)
    
    # 5. Populate search_vector for existing rows
    # This is a one-time backfill for existing data
    op.execute("""
        UPDATE memory_metadata
        SET search_vector = 
            setweight(to_tsvector('simple', COALESCE(content_preview, '')), 'A') ||
            setweight(to_tsvector('simple', array_to_string(COALESCE(tags, ARRAY[]::text[]), ' ')), 'D')
        WHERE search_vector IS NULL;
    """)
    
    # 6. Add NOT NULL constraint after backfill
    op.execute("""
        ALTER TABLE memory_metadata
        ALTER COLUMN search_vector SET NOT NULL;
    """)


def downgrade() -> None:
    """
    Remove full-text search support.
    
    This reverses all changes made in upgrade():
    - Drops trigger
    - Drops trigger function
    - Drops GIN index
    - Drops search_vector column
    """
    
    # Drop in reverse order of creation
    op.execute("DROP TRIGGER IF EXISTS memory_metadata_search_vector_trigger ON memory_metadata;")
    op.execute("DROP FUNCTION IF EXISTS memory_metadata_search_vector_update();")
    op.execute("DROP INDEX IF EXISTS idx_memory_metadata_search_vector;")
    op.execute("ALTER TABLE memory_metadata DROP COLUMN IF EXISTS search_vector;")
