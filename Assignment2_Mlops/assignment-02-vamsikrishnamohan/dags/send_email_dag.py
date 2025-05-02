from airflow import DAG
from airflow.providers.smtp.operators.email import EmailOperator
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.sensors.filesystem import FileSensor
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime, timedelta

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 2, 8),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "send_email_notification",
    default_args=default_args,
    schedule_interval=None,
    catchup=False
)

def check_new_news():
    pg_hook = PostgresHook(postgres_conn_id="postgres_default")
    sql = "SELECT headline, url FROM headlines WHERE scrape_time >= NOW() - INTERVAL '1 hour';"
    return pg_hook.get_records(sql)

def send_news_email():
    new_news = check_new_news()
    if not new_news:
        return "No new articles."

    news_content = "\n".join([f"- {title} ({url})" for title, url in new_news])
    email_body = f"New articles found:\n\n{news_content}"

    email_task = EmailOperator(
        task_id="send_email",
        to="krishvamsi321@gmai.com",
        subject="New Google News Articles",
        html_content=email_body,
        smtp_conn_id="smtp_default",
        dag=dag
    )
    email_task.execute(context={})

wait_for_status = FileSensor(
    task_id="wait_for_status",
    filepath="/usr/local/airflow/dags/run/status",
    poke_interval=30,
    timeout=600,
    mode="poke",
    dag=dag
)

delete_status_file = BashOperator(
    task_id="delete_status_file",
    bash_command="rm /usr/local/airflow/dags/run/status",
    dag=dag
)

wait_for_status >> send_news_email >> delete_status_file
