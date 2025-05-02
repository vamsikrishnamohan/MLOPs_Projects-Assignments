import os
import time
import logging
import hashlib
from datetime import datetime
import requests
import feedparser
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('rss_reader')

# Load environment variables
load_dotenv()

# Constants from environment variables
RSS_FEED_URL = os.getenv('RSS_FEED_URL', 'https://www.thehindu.com/news/national/?service=rss')
POLL_INTERVAL = int(os.getenv('POLL_INTERVAL', 600))  # 10 minutes in seconds
IMAGES_DIR = os.getenv('IMAGES_DIR', '/app/images')

# Database connection parameters
DB_HOST = os.getenv('DB_HOST', 'db')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('POSTGRES_DB', 'news_db')
DB_USER = os.getenv('POSTGRES_USER', 'postgres')
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'postgres')

# RSS Field paths from environment variables
TITLE_PATH = os.getenv('TITLE_PATH', 'title')
SUMMARY_PATH = os.getenv('SUMMARY_PATH', 'summary')
LINK_PATH = os.getenv('LINK_PATH', 'link')
TAGS_PATH = os.getenv('TAGS_PATH', 'tags')
PUBLISHED_PATH = os.getenv('PUBLISHED_PATH', 'published')
MEDIA_PATH = os.getenv('MEDIA_PATH', 'media_content')

# Ensure images directory exists
os.makedirs(IMAGES_DIR, exist_ok=True)

def get_db_connection():
    """Create a database connection"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME, 
            user=DB_USER,
            password=DB_PASSWORD
        )
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None

def parse_timestamp(timestamp_str):
    """Parse the RSS feed timestamp into a Python datetime object"""
    try:
        dt = datetime.strptime(timestamp_str, '%a, %d %b %Y %H:%M:%S %z')
        return dt
    except Exception as e:
        logger.warning(f"Failed to parse timestamp '{timestamp_str}': {e}")
        return datetime.now()

def download_image(image_url):
    """Download an image from a URL and save it locally"""
    if not image_url:
        return None
    
    try:
        # Create a filename based on the URL
        filename = hashlib.md5(image_url.encode()).hexdigest() + '.jpg'
        filepath = os.path.join(IMAGES_DIR, filename)
        
        # Skip download if the file already exists
        if os.path.exists(filepath):
            return filepath
            
        # Download the image
        response = requests.get(image_url, timeout=10)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            logger.info(f"Downloaded image to {filepath}")
            return filepath
        else:
            logger.warning(f"Failed to download image, status code: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error downloading image {image_url}: {e}")
        return None

def extract_image_url(entry):
    """Extract image URL from the entry's media content"""
    try:
        if MEDIA_PATH in entry and entry[MEDIA_PATH]:
            for media in entry[MEDIA_PATH]:
                if 'url' in media:
                    return media['url']
    except Exception as e:
        logger.warning(f"Error extracting image URL: {e}")
    return None

def extract_tags(entry):
    """Extract tags from the entry"""
    tags = []
    try:
        if TAGS_PATH in entry and entry[TAGS_PATH]:
            for tag in entry[TAGS_PATH]:
                if 'term' in tag:
                    tags.append(tag['term'])
    except Exception as e:
        logger.warning(f"Error extracting tags: {e}")
    return tags

def process_feed():
    """Fetch and process the RSS feed"""
    logger.info(f"Fetching RSS feed from {RSS_FEED_URL}")
    
    try:
        feed = feedparser.parse(RSS_FEED_URL)
        
        if not feed.entries:
            logger.warning("No entries found in the RSS feed")
            return
            
        logger.info(f"Found {len(feed.entries)} entries in the RSS feed")
        
        conn = get_db_connection()
        if not conn:
            return
            
        cursor = conn.cursor()
        
        for entry in feed.entries:
            try:
                # Extract required fields
                title = entry.get(TITLE_PATH, "")
                if not title:
                    logger.warning("Skipping entry with blank title")
                    continue
                    
                web_link = entry.get(LINK_PATH, "")
                if not web_link:
                    logger.warning(f"Skipping entry '{title}' with blank web link")
                    continue
                
                # Check if this article already exists in database
                cursor.execute(
                    "SELECT id FROM news WHERE web_link = %s",
                    (web_link,)
                )
                if cursor.fetchone():
                    logger.info(f"Article '{title}' already exists in database, skipping")
                    continue
                
                # Process other fields
                published_str = entry.get(PUBLISHED_PATH, "")
                published_date = parse_timestamp(published_str) if published_str else datetime.now()
                
                summary = entry.get(SUMMARY_PATH, "")
                
                tags = extract_tags(entry)
                
                # Handle image
                image_url = extract_image_url(entry)
                image_path = download_image(image_url) if image_url else None
                
                # Insert into database
                cursor.execute(
                    """
                    INSERT INTO news (title, publication_timestamp, web_link, image_path, tags, summary)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (title, published_date, web_link, image_path, tags, summary)
                )
                conn.commit()
                
                logger.info(f"Inserted article: '{title}'")
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Error processing entry: {e}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error fetching or processing feed: {e}")

def main():
    """Main function to poll the RSS feed at regular intervals"""
    logger.info("Starting RSS reader application")
    
    while True:
        try:
            process_feed()
            logger.info(f"Sleeping for {POLL_INTERVAL} seconds before next poll")
            time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            logger.info("Application stopped by user")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            time.sleep(60)  # Wait a minute before retrying on error

if __name__ == "__main__":
    main()