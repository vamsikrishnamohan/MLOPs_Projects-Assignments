#!/bin/bash
set -e

# Log function
log_info() {
    echo "INFO: $1"
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -c "INSERT INTO logs(log_level, message) VALUES ('INFO', '$1');"
}

log_error() {
    echo "ERROR: $1"
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -c "INSERT INTO logs(log_level, message) VALUES ('ERROR', '$1');"
}

# Check if tables exist
check_tables() {
    log_info "Checking if required tables exist..."
    
    # Check articles table
    ARTICLES_EXISTS=$(psql -tAc "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'articles');" --username "$POSTGRES_USER" --dbname "$POSTGRES_DB")
    
    # Check tags table
    TAGS_EXISTS=$(psql -tAc "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'tags');" --username "$POSTGRES_USER" --dbname "$POSTGRES_DB")
    
    # Check article_tags table
    ARTICLE_TAGS_EXISTS=$(psql -tAc "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'article_tags');" --username "$POSTGRES_USER" --dbname "$POSTGRES_DB")
    
    # Check logs table
    LOGS_EXISTS=$(psql -tAc "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'logs');" --username "$POSTGRES_USER" --dbname "$POSTGRES_DB")
    
    # If any table is missing, return false
    if [[ "$ARTICLES_EXISTS" != "t" || "$TAGS_EXISTS" != "t" || "$ARTICLE_TAGS_EXISTS" != "t" || "$LOGS_EXISTS" != "t" ]]; then
        return 1
    fi
    
    return 0
}

# Main validation
main() {
    if check_tables; then
        log_info "All required tables exist. Database is ready!"
    else
        log_error "One or more required tables are missing. Re-initializing database..."
        
        # Re-run the SQL script to create tables
        psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -f /docker-entrypoint-initdb.d/01-create-tables.sql
        
        # Check again after re-initialization
        if check_tables; then
            log_info "Database re-initialization successful!"
        else
            log_error "Database re-initialization failed! Container will exit."
            exit 1
        fi
    fi
}

# Execute main function
main