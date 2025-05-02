-- Create the news table if it doesn't exist
CREATE TABLE IF NOT EXISTS news (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    publication_timestamp TIMESTAMP NOT NULL,
    web_link TEXT NOT NULL,
    image_path TEXT,
    tags TEXT[],
    summary TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for faster querying by date
CREATE INDEX IF NOT EXISTS idx_news_publication_timestamp ON news(publication_timestamp);