"""
Test script with 5 example questions demonstrating the HR AI Assistant.

Run: python test_agent.py
"""

import asyncio
import logging
import sys
import io

# Fix Windows console encoding for Unicode characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)

from src.ingest import ingest_policies
from src.agent import ask_agent


# -- Example questions --------------------------------------------------------
EXAMPLES = [
    {
        "employee_id": "EMP001",
        "question": "How many leave days do I have left?",
        "description": "Structured data query - personal leave balance",
    },
    {
        "employee_id": "EMP005",
        "question": "What is the carry-over limit for unused annual leave?",
        "description": "RAG query - policy-only question about leave rules",
    },
    {
        "employee_id": "EMP013",
        "question": "Am I eligible for remote work?",
        "description": "Combined query - needs employee status (Probation) + remote work policy",
    },
    {
        "employee_id": "EMP003",
        "question": "How much training budget do I have remaining, and what can I spend it on?",
        "description": "Combined query - needs employee data + training policy",
    },
    {
        "employee_id": "EMP001",
        "question": "What is the stock price of AlNoor Technologies?",
        "description": "Out-of-scope question - should respond with 'I don't know'",
    },
]


async def run_examples():
    """Run all example questions and print results."""
    print("\n" + "=" * 70)
    print("  HR AI Assistant - Example Questions")
    print("=" * 70)

    # Ensure policies are ingested
    ingest_policies()

    for i, example in enumerate(EXAMPLES, 1):
        print(f"\n{'-' * 70}")
        print(f"  Question {i}: {example['description']}")
        print(f"{'-' * 70}")
        print(f"  Employee:  {example['employee_id']}")
        print(f"  Question:  {example['question']}")
        print()

        try:
            result = await ask_agent(
                employee_id=example["employee_id"],
                question=example["question"],
            )
            print(f"  Source:    {result['source']}")
            if result.get('references'):
                for ref in result['references']:
                    print(f"  Ref:      {ref['policy_name']} S{ref['section_number']}: {ref['section_title']} ({ref['relevance_score']:.0%})")
            print(f"  Answer:    {result['answer'][:500]}")
        except Exception as e:
            print(f"  ERROR:     {e}")

        print()

    print("=" * 70)
    print("  All examples completed.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_examples())
