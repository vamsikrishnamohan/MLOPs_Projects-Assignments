import hashlib

def generate_hash(headline, article_url, article_date):
    """Generate a unique hash based on headline, URL, and date."""
    return hashlib.sha256(f"{headline}|{article_url}|{article_date}".encode()).hexdigest()

def is_duplicate(story, collection):
    """Check if the story already exists in the database."""
    news_hash = generate_hash(story["headline"], story["url"], story["article_date"])
    
    return collection.find_one({"news_hash": news_hash}) is not None
