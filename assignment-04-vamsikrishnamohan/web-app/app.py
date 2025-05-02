import os
import logging
from datetime import datetime, timedelta
from flask import Flask, render_template, request, send_file, url_for
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('web_app')

# Load environment variables
load_dotenv()

# Constants from environment variables
DB_HOST = os.getenv('DB_HOST', 'db')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('POSTGRES_DB', 'news_db')
DB_USER = os.getenv('POSTGRES_USER', 'postgres')
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'postgres')
IMAGES_DIR = os.getenv('IMAGES_DIR', '/app/images')

app = Flask(__name__)

def get_db_connection():
    """Create a database connection"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME, 
            user=DB_USER,
            password=DB_PASSWORD,
            cursor_factory=RealDictCursor
        )
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve image files"""
    try:
        return send_file(os.path.join(IMAGES_DIR, filename))
    except Exception as e:
        logger.error(f"Error serving image {filename}: {e}")
        return "Image not found", 404

@app.route('/')
def index():
    """Home page - show news articles with date filter"""
    try:
        # Get date filter from query parameters, default to today
        filter_date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
        
        try:
            # Validate date format
            filter_datetime = datetime.strptime(filter_date, '%Y-%m-%d')
            next_day = filter_datetime + timedelta(days=1)
        except ValueError:
            logger.warning(f"Invalid date format: {filter_date}")
            filter_datetime = datetime.now()
            next_day = filter_datetime + timedelta(days=1)
            filter_date = filter_datetime.strftime('%Y-%m-%d')
        
        conn = get_db_connection()
        if not conn:
            return render_template('index.html', 
                                  news=[], 
                                  date=filter_date,
                                  error="Database connection failed")
        
        cursor = conn.cursor()
        
        # Query news articles for the selected date
        cursor.execute(
            """
            SELECT title, publication_timestamp, web_link, image_path, tags, summary 
            FROM news 
            WHERE publication_timestamp >= %s AND publication_timestamp < %s
            ORDER BY publication_timestamp DESC
            """,
            (filter_datetime, next_day)
        )
        
        news_articles = cursor.fetchall()
        
        # Process image paths to be served through Flask
        for article in news_articles:
            if article['image_path']:
                # Extract just the filename from the full path
                article['image_filename'] = os.path.basename(article['image_path'])
            else:
                article['image_filename'] = None
        
        cursor.close()
        conn.close()
        
        return render_template('index.html', 
                              news=news_articles, 
                              date=filter_date)
                              
    except Exception as e:
        logger.error(f"Error rendering index page: {e}")
        return render_template('index.html', 
                              news=[], 
                              date=datetime.now().strftime('%Y-%m-%d'),
                              error=f"An error occurred: {e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)