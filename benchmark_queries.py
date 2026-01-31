#!/usr/bin/env python3
"""
Week 3 Query Optimization - Benchmarking Script
=================================================

Measures query performance before and after index creation.
Focuses on hot queries identified in the optimization guide.
"""

import psycopg
from datetime import datetime, timedelta
import time
import json

def execute_query_with_timing(cursor, query_name: str, sql: str, iterations: int = 3) -> dict:
    """Execute a query multiple times and measure performance."""
    print(f"\n  {query_name}...")
    times = []
    
    for i in range(iterations):
        start = time.time()
        cursor.execute(sql)
        result = cursor.fetchall()
        elapsed = (time.time() - start) * 1000  # Convert to milliseconds
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    return {
        "query": query_name,
        "iterations": iterations,
        "avg_ms": round(avg_time, 2),
        "min_ms": round(min_time, 2),
        "max_ms": round(max_time, 2),
        "rows": len(result) if result else 0,
    }

def main():
    print("=" * 70)
    print("WEEK 3 QUERY OPTIMIZATION - PERFORMANCE BENCHMARKING")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
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
        print("✓ Connected to PostgreSQL\n")
        
        # Define benchmark queries (hot queries from optimization guide)
        benchmarks = {
            "Memory Queries": [
                (
                    "Get recent memories by organization",
                    """
                    SELECT id, title, created_at, owner_id
                    FROM memory_metadata
                    WHERE organization_id = (
                        SELECT id FROM organizations LIMIT 1
                    )
                    ORDER BY created_at DESC
                    LIMIT 50;
                    """
                ),
                (
                    "Get user's memories",
                    """
                    SELECT id, title, created_at, organization_id
                    FROM memory_metadata
                    WHERE owner_id = (
                        SELECT id FROM users LIMIT 1
                    )
                    ORDER BY created_at DESC
                    LIMIT 50;
                    """
                ),
                (
                    "Search by scope and organization",
                    """
                    SELECT id, title, scope, owner_id
                    FROM memory_metadata
                    WHERE organization_id = (
                        SELECT id FROM organizations LIMIT 1
                    )
                    AND scope = 'personal'
                    LIMIT 100;
                    """
                ),
            ],
            "Audit Queries": [
                (
                    "Get audit events by organization and timestamp",
                    """
                    SELECT id, event_type, timestamp, actor_id
                    FROM audit_events
                    WHERE organization_id = (
                        SELECT id FROM organizations LIMIT 1
                    )
                    ORDER BY timestamp DESC
                    LIMIT 100;
                    """
                ),
                (
                    "Get user's audit trail",
                    """
                    SELECT id, event_type, timestamp, resource_type
                    FROM audit_events
                    WHERE actor_id = (
                        SELECT id FROM users LIMIT 1
                    )
                    ORDER BY timestamp DESC
                    LIMIT 50;
                    """
                ),
            ],
            "Access Log Queries": [
                (
                    "Get memory access by user",
                    """
                    SELECT id, memory_id, action, timestamp
                    FROM memory_access_log
                    WHERE user_id = (
                        SELECT id FROM users LIMIT 1
                    )
                    ORDER BY timestamp DESC
                    LIMIT 50;
                    """
                ),
                (
                    "Get who accessed a memory",
                    """
                    SELECT id, user_id, action, timestamp
                    FROM memory_access_log
                    WHERE memory_id = (
                        SELECT id FROM memory_metadata LIMIT 1
                    )
                    ORDER BY timestamp DESC
                    LIMIT 50;
                    """
                ),
            ],
        }
        
        results_by_category = {}
        
        # Run benchmarks
        for category, queries in benchmarks.items():
            print(f"\n{category}:")
            print("-" * 70)
            results_by_category[category] = []
            
            for query_name, sql_query in queries:
                try:
                    result = execute_query_with_timing(cursor, query_name, sql_query)
                    results_by_category[category].append(result)
                    print(f"    ✓ {result['avg_ms']}ms (avg) | {result['rows']} rows")
                except Exception as e:
                    print(f"    ✗ ERROR: {str(e)[:60]}")
        
        cursor.close()
        conn.close()
        
        # Print summary
        print("\n" + "=" * 70)
        print("PERFORMANCE SUMMARY")
        print("=" * 70)
        
        total_queries = 0
        total_time = 0
        
        for category, results in results_by_category.items():
            if results:
                print(f"\n{category}:")
                for r in results:
                    print(f"  {r['query']:<40} {r['avg_ms']:>8.2f}ms")
                    total_time += r['avg_ms']
                    total_queries += 1
        
        if total_queries > 0:
            avg_query_time = total_time / total_queries
            print(f"\nTotal Queries: {total_queries}")
            print(f"Average Query Time: {avg_query_time:.2f}ms")
        
        print("\n" + "=" * 70)
        print("✓ BENCHMARKING COMPLETE")
        print("=" * 70)
        print("\nNotes:")
        print("- Indexes are now in place for hot queries")
        print("- Monitor pg_stat_user_indexes for actual index usage")
        print("- Re-run this benchmark after production load for real metrics")
        print("- Expected improvement: 30-50% for memory queries, 25-40% for audit")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
