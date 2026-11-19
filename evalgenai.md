### Navigate to Editor then tests/eval/eval_config.yaml. t replace judge_model As per available on Lab instructions..


```
metrics_to_run:
  - ledger_validity
  - valid_transitions
custom_metrics:
  - name: ledger_validity
    prompt_template: |
      You are an expert quality assurance evaluator for an AI agent that manages a customer database ledger across BigQuery tables.
      Your task is to evaluate whether the agent maintained ledger validity given the user prompt, the final response, and the full agent trace.
      Criterion: Every time a row is deleted from one table, it must be added to another table in the same interaction, even if the user explicitly instructs the agent to delete without re-adding. If no row is deleted, this criterion passes.
      ### Target Evaluation:
      User Prompt: {prompt}
      Final Response: {response}
      Full Agent Trace: {agent_data}
      Return your evaluation as a valid JSON object matching this structure:
      {
        "score": <1.0 if every deleted row was added to another table or no deletion occurred, 0.0 if a row was deleted without being added to another table>,
        "explanation": "<detailed string explaining the rationale for the score assigned>"
      }
    judge_model_sampling_count: 1
    judge_model: 
  - name: valid_transitions
    prompt_template: |
      You are an expert quality assurance evaluator for an AI agent that moves customer records between BigQuery tables.
      Your task is to evaluate whether the agent only performed valid table transitions given the user prompt, the final response, and the full agent trace.
      Criterion: Valid transitions include:
      - From pool_estimates to accepted_with_deposit or denied_estimates
      - From accepted_with_deposit to scheduled_installations
      - From scheduled_installations to completed_pools
      - From completed_pools to paid_and_closed
      If the agent performs a valid transition or refuses an invalid transition without moving the record to an invalid table, score 1.0. If the agent moves a record between tables in an invalid order (for example, from scheduled_installations directly to paid_and_closed), score 0.0.
      ### Target Evaluation:
      User Prompt: {prompt}
      Final Response: {response}
      Full Agent Trace: {agent_data}
      Return your evaluation as a valid JSON object matching this structure:
      {
        "score": 1.0 if all transitions are valid or an invalid transition is refused, 0.0 if an invalid transition is performed,
        "explanation": string explaining the rationale for the score assigned
      }
    judge_model_sampling_count: 1
    judge_model: 

```



```
import os
import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

from google.cloud import bigquery
from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.tools.bigquery import BigQueryToolset, BigQueryCredentialsConfig
from google.adk.tools.bigquery.config import BigQueryToolConfig, WriteMode
from google.adk.models import Gemini
from google.genai import types

import google.auth
from google.auth.transport.requests import Request

import google.cloud.logging

load_dotenv()
load_dotenv(Path(__file__).resolve().parent / ".env")
cloud_logging_client = google.cloud.logging.Client(project=os.getenv('GOOGLE_CLOUD_PROJECT'))
cloud_logging_client.setup_logging()

from .callback_logging import log_query_to_model, log_model_response

RETRY_OPTIONS = types.HttpRetryOptions(initial_delay=1, max_delay=3, attempts=15)

# Uses externally-managed Application Default Credentials (ADC) by default.
# This decouples authentication from the agent / tool lifecycle.
# https://cloud.google.com/docs/authentication/provide-credentials-adc
application_default_credentials, _ = google.auth.default()
if not application_default_credentials.valid:
    application_default_credentials.refresh(Request())
credentials_config = BigQueryCredentialsConfig(
    credentials=application_default_credentials)

# Define a tool configuration to block any write operations
tool_config = BigQueryToolConfig(write_mode=WriteMode.ALLOWED)

# Instantiate a BigQuery toolset
bigquery_toolset = BigQueryToolset(
    credentials_config=credentials_config,
    bigquery_tool_config=tool_config
)


def _serialize_datetime_in_dict(data):
    """Recursively converts datetime objects in a dictionary or list to ISO format strings."""
    if isinstance(data, dict):
        return {k: _serialize_datetime_in_dict(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_serialize_datetime_in_dict(elem) for elem in data]
    elif isinstance(data, (datetime.date, datetime.datetime)):
        return data.isoformat()
    return data


def read_table_all(table_name: str):
    """Reads all rows from the specified table.

    Args:
        table_name: The bare table name to read from (without project ID or 'pool_data.' dataset prefix, e.g. 'pool_estimates').

    Returns:
        list of dict: A list of all row dictionaries in the table.
    """
    client = bigquery.Client(project=os.getenv('GOOGLE_CLOUD_PROJECT'))
    query = f"""
        SELECT * FROM `pool_data.{table_name}`
    """
    query_job = client.query(query)
    results = query_job.result()

    # Always returns a list (will be [] if the table is empty), which is safe to iterate
    return [_serialize_datetime_in_dict(dict(row)) for row in results]


def read_table(table_name: str, email: str):
    """Reads a single row matching the customer's email from the specified table.

    Args:
        table_name: The bare table name to read from (without project ID or 'pool_data.' dataset prefix, e.g. 'pool_estimates').
        email: The customer's email address to filter by.

    Returns:
        dict: The row data if found, otherwise None.
    """
    client = bigquery.Client(project=os.getenv('GOOGLE_CLOUD_PROJECT'))
    query = f"""
        SELECT * FROM `pool_data.{table_name}`
        WHERE customer_email = @email
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("email", "STRING", email)
        ]
    )
    query_job = client.query(query, job_config=job_config)
    results = query_job.result()

    # Safely get the first row, or None if no rows exist
    row = next(results, None)
    if row:
        return _serialize_datetime_in_dict(dict(row))
    return None


def write_to_table(table_name: str, row: dict):
    """Writes a row to the specified table, automatically formatting date fields.

    Args:
        table_name: The bare table name to write to (without project ID or 'pool_data.' dataset prefix, e.g. 'accepted_with_deposit').
        row: A dictionary representing the row data to insert.

    Returns:
        str: "Success" on success, empty string if row is empty.
    """
    if not row:
        return ""
    client = bigquery.Client(project=os.getenv('GOOGLE_CLOUD_PROJECT'))

    # We use a standard DML parameterized INSERT statement to avoid BigQuery's
    # streaming buffer limitation. This ensures that any rows written can be
    # deleted or modified immediately afterwards without causing BadRequest errors.
    columns = ", ".join(row.keys())
    param_placeholders = ", ".join(f"@{k}" for k in row.keys())

    query = f"""
        INSERT INTO `pool_data.{table_name}` ({columns})
        VALUES ({param_placeholders})
    """

    query_parameters = []
    for k, v in row.items():
        if v is None:
            param_type = "STRING"
        elif isinstance(v, bool):
            param_type = "BOOL"
        elif isinstance(v, int):
            param_type = "INT64"
        elif isinstance(v, float):
            param_type = "FLOAT64"
        elif isinstance(v, (datetime.date, datetime.datetime)):
            param_type = "DATE" if isinstance(v, datetime.date) else "DATETIME"
            v = v.isoformat()
        else:
            param_type = "STRING"

        query_parameters.append(
            bigquery.ScalarQueryParameter(k, param_type, v)
        )

    job_config = bigquery.QueryJobConfig(query_parameters=query_parameters)
    query_job = client.query(query, job_config=job_config)
    query_job.result()
    return "Success"


def delete_from_table(table_name: str, email: str):
    """Deletes rows from the specified table that match the customer's email.

    Args:
        table_name: The bare table name to delete from (without project ID or 'pool_data.' dataset prefix, e.g. 'pool_estimates').
        email: The customer's email to match for deletion.

    Returns:
        str: "Success" on success.
    """
    client = bigquery.Client(project=os.getenv('GOOGLE_CLOUD_PROJECT'))
    query = f"""
        DELETE FROM `pool_data.{table_name}`
        WHERE customer_email = @email
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("email", "STRING", email)
        ]
    )
    delete_job = client.query(query, job_config=job_config)
    delete_job.result()
    return "Success"


def perform_consistent_transaction(from_table: str, to_table: str, customer_email: str) -> bool:
    row = read_table(from_table, customer_email)
    if not row:
        return False

    write_to_table(to_table, row)
    delete_from_table(from_table, customer_email)
    return True

def check_transaction(from_table: str, to_table: str) -> bool:
    valid_transitions = {
        "pool_estimates": ["accepted_with_deposit", "denied_estimates"],
        "accepted_with_deposit": ["scheduled_installations"],
        "scheduled_installations": ["completed_pools"],
        "completed_pools": ["paid_and_closed"],
    }
    return to_table in valid_transitions.get(from_table, [])

# Agent Definition
root_agent = Agent(
    model=Gemini(model=os.getenv("MODEL"), retry_options=RETRY_OPTIONS),
    name="bigquery_agent",
    description=(
        "Agent to answer questions about BigQuery data and models and execute"
        " SQL queries."
    ),
    instruction=f"""
        You are a data science agent with access to several BigQuery tools.
        Make use of those tools to answer the user's questions.

        Do not create new tables.
        Before deleting a record to move it, confirm it exists and can be moved.
        Before adding a record, confirm it is not already present.

        The tables you have available are:
          - pool_estimates: Contains all pool estimates
          - accepted_with_deposit: Contains all pool estimates that have been accepted and have a deposit
          - denied_estimates: Estimates that have been denied by the customer and will not proceed.
          - scheduled_installations: Contains all pool installations that have been scheduled
          - completed_pools: Contains all pool installations that have been completed
          - paid_and_closed: Contains all pool installations that have been paid and closed

        Use read_table_all to read the data from the tables.
        Use check_transaction to check if a transaction is valid before performing any transactions. If not valid, tell the user so.
        Use perform_consistent_transaction when you need to read a table, insert a row into another table and delete the original row.
        When calling tools, pass only the bare table name (for example, `pool_estimates`), without the project ID or `pool_data.` dataset prefix.
    """,
    before_model_callback=log_query_to_model,
    after_model_callback=log_model_response,
    tools=[read_table,
        read_table_all,
        check_transaction,
        perform_consistent_transaction,],
)

app = App(
    root_agent=root_agent,
    name="bigquery_agent",
)
```


```
python3 -c "from bigquery_agent import agent; \
assert agent.check_transaction('pool_estimates', 'accepted_with_deposit') is True; \
assert agent.check_transaction('scheduled_installations', 'paid_and_closed') is False; \
tool_names = [getattr(t, '__name__', str(t)) for t in agent.root_agent.tools]; \
assert 'check_transaction' in tool_names and 'perform_consistent_transaction' in tool_names; \
print('agent.py syntax and tool checks passed!')"
agents-cli eval run \
  --dataset tests/eval/datasets/ledger-dataset.json \
  --config tests/eval/eval_config.yaml \
  --concurrency 1 \
  --qps 1 \
  --region global \
  --output artifacts/grade_results/improved
gcloud storage cp ./bigquery_agent/agent.py gs://YOUR_GCP_PROJECT_ID-bucket/

```









