"""
Structured data tool: queries the employee CSV to answer questions about
specific employee information like leave balance, department, hire date,
remote model, performance rating, and training budget.
"""

import logging

import pandas as pd
from langchain_core.tools import tool

from src.config import EMPLOYEES_CSV

logger = logging.getLogger(__name__)

# ── Load employee data at module level ───────────────────────────────────────
_df: pd.DataFrame | None = None

# Store the last retrieved employee for the API to access as a reference
_last_references: list[dict] = []

def get_last_references() -> list[dict]:
    """Get the references from the last employee query."""
    return list(_last_references)


def _get_dataframe() -> pd.DataFrame:
    """Lazy-load the employee CSV into a pandas DataFrame."""
    global _df
    if _df is None:
        _df = pd.read_csv(EMPLOYEES_CSV)
        logger.info(f"Loaded {len(_df)} employees from {EMPLOYEES_CSV}")
    return _df


@tool
def query_employee_data(employee_id: str) -> str:
    """Look up personal employee information from the HR database including
    leave balance, department, grade level, hire date, remote work model,
    manager, performance rating, training budget, and employment status.

    Use this tool when the question is about a SPECIFIC employee's personal
    data, such as their leave balance, department, manager, or performance
    rating.

    Args:
        employee_id: The employee's ID (e.g., "EMP001").

    Returns:
        A formatted summary of the employee's data, or an error message
        if the employee is not found.
    """
    global _last_references
    df = _get_dataframe()

    # Normalize the ID for matching
    emp_id = employee_id.strip().upper()
    match = df[df["employee_id"].str.upper() == emp_id]

    if match.empty:
        _last_references = []
        return (
            f"Employee '{employee_id}' not found in the database. "
            f"Valid IDs range from EMP001 to EMP{len(df):03d}."
        )

    row = match.iloc[0]

    # Save as a reference for the UI citation
    _last_references = [{
        "policy_name": "Employee Record",
        "section_number": row["employee_id"],
        "section_title": row["full_name"],
        "source_file": "employees.csv",
        "relevance_score": 1.0,
    }]

    # Calculate remaining training budget
    training_remaining = row["training_budget_sar"] - row["training_spent_sar"]

    return (
        f"Employee Record for {row['full_name']} ({row['employee_id']}):\n"
        f"  • Department: {row['department']}\n"
        f"  • Grade Level: {row['grade_level']}\n"
        f"  • Hire Date: {row['hire_date']}\n"
        f"  • Employment Status: {row['employment_status']}\n"
        f"  • Manager: {row['manager']}\n"
        f"  • Remote Work Model: {row['remote_model']}\n"
        f"  • Annual Leave Entitlement: {row['annual_leave_days']} days\n"
        f"  • Leave Taken: {row['leave_taken']} days\n"
        f"  • Leave Balance: {row['leave_balance']} days remaining\n"
        f"  • Performance Rating (2024): {row['performance_rating_2024']} / 5\n"
        f"  • Training Budget: {row['training_budget_sar']:,.0f} SAR\n"
        f"  • Training Spent: {row['training_spent_sar']:,.0f} SAR\n"
        f"  • Training Remaining: {training_remaining:,.0f} SAR"
    )
