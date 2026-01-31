#!/usr/bin/env python3
"""
Week 3 Query Optimization - Index Creation Script
==================================================

Creates composite indexes on hot tables for performance improvement.
"""

import psycopg
from datetime import datetime
import sys

def main():
    print("=" * 70)
    print("WEEK 3 QUERY OPTIMIZATION - Creating Database Indexes")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Target: PostgreSQL at localhost:5432 (database: ninai)")
    print()
    
    try:
        # Connect to PostgreSQL
        conn = psycopg.connect(
            host="localhost",
            user="ninai",
            password="ninai_dev_password",
            dbname="ninai",
        )
        
        cursor = conn.cursor()
        print("✓ Connected to PostgreSQL")
        print()
        
        # Define indexes to create (using actual table names and columns)
        indexes = [
            ("ix_memory_metadata_org_created", """
                CREATE INDEX CONCURRENTLY ix_memory_metadata_org_created
                ON memory_metadata(organization_id, created_at DESC);
            """),
            ("ix_memory_metadata_owner_created", """
                CREATE INDEX CONCURRENTLY ix_memory_metadata_owner_created
                ON memory_metadata(owner_id, created_at DESC);
            """),
            ("ix_memory_metadata_org_scope", """
                CREATE INDEX CONCURRENTLY ix_memory_metadata_org_scope
                ON memory_metadata(organization_id, scope);
            """),
            ("ix_audit_events_actor_time", """
                CREATE INDEX CONCURRENTLY ix_audit_events_actor_time
                ON audit_events(actor_id, timestamp DESC);
            """),
            ("ix_memory_access_log_user_ts", """
                CREATE INDEX CONCURRENTLY ix_memory_access_log_user_ts
                ON memory_access_log(user_id, timestamp DESC);
            """),
            ("ix_memory_access_log_memory_ts", """
                CREATE INDEX CONCURRENTLY ix_memory_access_log_memory_ts
                ON memory_access_log(memory_id, timestamp DESC);
            """),
            ("ix_memory_metadata_tags", """
                CREATE INDEX CONCURRENTLY ix_memory_metadata_tags
                ON memory_metadata USING GIN(tags);
            """),
        ]
        
        # Create indexes (CONCURRENTLY requires autocommit mode)
        conn.autocommit = True
        print("Creating indexes (non-blocking with CONCURRENTLY)...")
        print("-" * 70)
        created = 0
        skipped = 0
        failed = 0
        
        for idx_name, sql_stmt in indexes:
            try:
                cursor.execute(sql_stmt)
                print(f"✓ {idx_name:40} (created)")
                created += 1
            except psycopg.errors.DuplicateObject:
                print(f"⊘ {idx_name:40} (already exists)")
                skipped += 1
            except Exception as e:
                print(f"✗ {idx_name:40} ERROR: {str(e)[:40]}")
                failed += 1
        
        print("-" * 70)
        print(f"\nSummary: {created} created, {skipped} skipped, {failed} failed")
        print()
        
        # Verify indexes
        conn.autocommit = False
        print("=" * 70)
        print("INDEX VERIFICATION")
        print("=" * 70)
        
        cursor.execute("""
            SELECT 
              schemaname,
              relname,
              indexrelname,
              idx_scan,
              pg_size_pretty(pg_relation_size(indexrelid)) AS size
            FROM pg_stat_user_indexes
            WHERE indexrelname LIKE 'ix_memory_%'
               OR indexrelname LIKE 'ix_audit_%'
            ORDER BY relname, indexrelname;
        """)
        
        results = cursor.fetchall()
        if results:
            print(f"\n{len(results)} Indexes Status:\n")
            print(f"{'Table':<20} {'Index Name':<40} {'Size':<10}")
            print("-" * 70)
            for schemaname, tablename, indexname, idx_scan, size in results:
                print(f"{tablename:<20} {indexname:<40} {size:<10}")
        else:
            print("\nNo indexes found (they may be in creation progress)")
        
        # Check table sizes before index improvement
        print("\n" + "=" * 70)
        print("TABLE SIZES")
        print("=" * 70)
        
        cursor.execute("""
            SELECT 
              tablename,
              pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size,
              (SELECT count(*) FROM pg_class WHERE relname = tablename) AS rows
            FROM pg_tables
            WHERE schemaname = 'public'
              AND tablename IN ('memories', 'audit_events', 'capability_grants', 'memory_access_logs', 'users')
            ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
        """)
        
        results = cursor.fetchall()
        if results:
            print(f"\n{'Table':<25} {'Size':<15}")
            print("-" * 40)
            for tablename, size, rows in results:
                print(f"{tablename:<25} {size:<15}")
        
        cursor.close()
        conn.close()
        
        print("\n" + "=" * 70)
        print("✓ INDEX CREATION COMPLETE")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Wait for index creation to complete (if running CONCURRENTLY)")
        print("2. Run query benchmarks to verify performance improvement")
        print("3. Monitor pg_stat_user_indexes for usage patterns")
        print()
        
        return 0
        
    except psycopg.OperationalError as e:
        print(f"\n✗ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("- Verify PostgreSQL is running on localhost:5432")
        print("- Check credentials: user=ninai, password=ninai_dev_password")
        print("- Check database exists: ninai")
        return 1
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
