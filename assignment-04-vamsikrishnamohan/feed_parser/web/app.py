from flask import Flask, render_template, request
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from datetime import datetime
import time

app = Flask(__name__)

# Database connection with retry logic
def get_db_connection():
    retry_count = 0
    max_retries = 5
    retry_delay = 5  # seconds

    while retry_count < max_retries:
        try:
            conn = psycopg2.connect(
                host=os.environ.get('DB_HOST', 'db'),
                database=os.environ.get('POSTGRES_DB', 'rssdb'),
                user=os.environ.get('POSTGRES_USER', 'rssuser'),
                password=os.environ.get('POSTGRES_PASSWORD', 'rsspass')
            )
            return conn
        except psycopg2.OperationalError as e:
            retry_count += 1
            if retry_count == max_retries:
                print(f"Failed to connect to database after {max_retries} attempts: {e}")
                raise
            print(f"Database connection attempt {retry_count} failed. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)

@app.route('/')
def index():
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(
        """
        SELECT a.*, s.name as source_name
        FROM articles a
        JOIN rss_sources s ON a.source_id = s.id
        WHERE DATE(a.published) = CURRENT_DATE
        ORDER BY a.published DESC
        """
    )
    articles = cur.fetchall()
    cur.close()
    conn.close()

    # Format articles for template
    formatted_articles = [(article['source_name'], {
        'title': article['title'],
        'link': article['link'],
        'published': article['published'].strftime('%Y-%m-%d %H:%M:%S')
    }) for article in articles]

    return render_template('index.html', articles=formatted_articles)

@app.route('/search')
def search():
    query = request.args.get('q', '')
    
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    # Search articles
    cur.execute(
        """
        SELECT a.*, s.name as source_name
        FROM articles a
        JOIN rss_sources s ON a.source_id = s.id
        WHERE a.title ILIKE %s
        ORDER BY a.published DESC
        """,
        (f'%{query}%',)
    )
    
    articles = cur.fetchall()
    cur.close()
    conn.close()
    
    # Format articles for template
    formatted_articles = [(article['source_name'], {
        'title': article['title'],
        'link': article['link'],
        'published': article['published'].strftime('%Y-%m-%d %H:%M:%S')
    }) for article in articles]
    
    return render_template('search_results.html', articles=formatted_articles, query=query)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)