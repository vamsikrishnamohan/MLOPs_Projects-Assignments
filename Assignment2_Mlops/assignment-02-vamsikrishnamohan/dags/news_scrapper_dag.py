from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
from scraper import get_top_stories
from storage import store_news

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "news_scraper_pipeline",
    default_args=default_args,
    schedule_interval="@hourly",
    catchup=False
)

def scrape_news():
    news_data = get_top_stories()
    return news_data

scrape_task = PythonOperator(
    task_id="scrape_news",
    python_callable=scrape_news,
    dag=dag
)

def store_news_task(**kwargs):
    ti = kwargs['ti']
    news_data = ti.xcom_pull(task_ids="scrape_news")
    store_news(news_data)

store_task = PythonOperator(
    task_id="store_news",
    python_callable=store_news_task,
    provide_context=True,
    dag=dag
)

touch_status_file = BashOperator(
    task_id="create_status_file",
    bash_command="echo 1 > /usr/local/airflow/dags/run/status",
    dag=dag
)


scrape_task >> store_task >> touch_status_file
