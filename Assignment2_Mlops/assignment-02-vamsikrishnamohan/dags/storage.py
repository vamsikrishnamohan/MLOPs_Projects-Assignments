from airflow.providers.postgres.hooks.postgres import PostgresHook

def store_news(news_data):
    """Stores news headlines & images in PostgreSQL."""
    pg_hook = PostgresHook(postgres_conn_id="postgres_default")

    for story in news_data:
        sql = """
        INSERT INTO headlines (headline, url, article_date, scrape_time) 
        VALUES (%s, %s, %s, %s) RETURNING id;
        """
        headline_id = pg_hook.get_first(sql, parameters=(story["headline"], story["url"], story["article_date"], story["scrape_time"]))

        if story["image"]:
            sql_image = "INSERT INTO news_images (headline_id, image) VALUES (%s, %s);"
            pg_hook.run(sql_image, parameters=(headline_id[0], story["image"]))
