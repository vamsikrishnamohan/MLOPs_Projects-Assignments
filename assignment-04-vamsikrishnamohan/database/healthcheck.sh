#!/bin/bash
set -e

# Get PostgreSQL connection info from environment variables
PGUSER="${POSTGRES_USER:-postgres}"
PGDATABASE="${POSTGRES_DB:-news_db}"

# Check database connection
pg_isready -U "$PGUSER" -d "$PGDATABASE" -h localhost -p 5432

# Check if the news table exists
psql -U "$PGUSER" -d "$PGDATABASE" -t -c "SELECT to_regclass('public.news')" | grep -q "news"

# Exit with success code if all checks passed
exit 0