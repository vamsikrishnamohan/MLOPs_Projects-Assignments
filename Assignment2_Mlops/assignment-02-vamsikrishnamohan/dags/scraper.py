import os
import time
import requests
import base64
from bs4 import BeautifulSoup
from datetime import datetime
import yaml

# Load config from YAML
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

google_news_url = config["google_news_url"]

def get_top_stories():
    """Scrape top stories from Google News using BeautifulSoup."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }
    response = requests.get(google_news_url, headers=headers)
    
    if response.status_code != 200:
        print("Failed to retrieve news:", response.status_code)
        return []
    
    soup = BeautifulSoup(response.text, "html.parser")
    articles = soup.select("article")
    
    print(f"Found {len(articles)} articles.")  # Debugging output
    
    data = []
    story_count = 0
    
    for article in articles:
        story_count += 1
        image_base64 = None

        # Extract headline
        headline_tag = article.select_one("h3")
        headline_text = headline_tag.text.strip() if headline_tag else "No headline"

        # Extract URL
        link_tag = article.select_one("a[href]")
        url = f"https://news.google.com{link_tag['href'][1:]}" if link_tag else "#"

        # Extract image (handle lazy loading and relative URLs)
        img_tag = article.select_one("img")
        img_url = img_tag["src"] if img_tag else None

        if img_url and img_url.startswith("/api/attachments"):
            img_url = f"https://news.google.com{img_url}"

        if img_url:
            try:
                response = requests.get(img_url, stream=True)
                if response.status_code == 200:
                    image_base64 = base64.b64encode(response.content).decode('utf-8')
            except Exception as e:
                print(f"Error encoding image: {e}")

        # Extract article date
        time_tag = article.select_one("time")
        article_date = time_tag["datetime"] if time_tag else "Unknown"

        metadata = {
            "headline": headline_text,
            "image_base64": image_base64,
            "url": url,
            "scrape_time": datetime.now().strftime("%H:%M:%S %Y-%m-%d"),
            "article_date": article_date,
            "story_count": story_count
        }
        
        data.append(metadata)
        time.sleep(1)  # Simulate scrolling delay
    
    return data

# Example usage
if __name__ == "__main__":
    news_data = get_top_stories()
    for news in news_data[:5]:  # Print first 5 stories
    
        print(news)

