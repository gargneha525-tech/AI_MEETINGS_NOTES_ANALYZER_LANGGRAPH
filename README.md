# AI Meeting Notes Analyzer Using LangGraph

## 1. Project Overview

The AI Meeting Notes Analyzer is an AI-powered application that
converts user-provided meeting transcripts into structured reports.

The application uses Python, LangGraph, LangChain, Ollama, Pydantic,
and Streamlit to automate meeting analysis.

Users can enter their own meeting transcripts and receive a summary,
key discussion topics, action items, task owners, and task priorities.

## 2. Project Objectives

The main objectives are:

- Generate concise meeting summaries.
- Extract key discussion topics.
- Identify agreed action items.
- Identify the person responsible for each task.
- Classify action items into priority levels.
- Explain the reasoning behind task priorities.
- Generate downloadable JSON and CSV reports.
- Demonstrate conditional routing using LangGraph.
- Allow users to submit their own meeting transcripts.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| LangGraph | Workflow orchestration and conditional routing |
| LangChain | Integration with the language model |
| Ollama | Local language model runtime |
| Pydantic | Structured output validation across all agents |
| Streamlit | Interactive user interface |
| Pandas | Tabular action-item presentation |
| python-dotenv | Optional environment configuration |

## 4. Project Structure

The project folder should contain:

    Capstone 3/
    |
    |-- app.py
    |-- workflow.py
    |-- README.md
    |-- .env                 (optional)
    |-- outputs/             (created automatically)
    |
    |-- cp3env/              (Python environment, if used)

Do not name the Streamlit application `app(2).py`.
Save it as `app.py` so the launch command works as documented.

## 5. System Architecture

The application uses the following workflow nodes:

1. Input Validation Node
   - Validates the transcript.
   - Rejects empty or excessively short inputs.

2. Topic Extraction Agent
   - Identifies the principal discussion topics using Pydantic structured output.
   - Removes duplicate topics.

3. Meeting Summary Agent
   - Produces a concise summary using Pydantic structured output.
   - Captures decisions, issues, and deadlines when available.

4. Action Item Extraction Agent
   - Identifies agreed tasks.
   - Extracts the responsible person.
   - Avoids inventing missing information.

5. Conditional Routing
   - Checks whether action items exist.
   - Routes the workflow accordingly.

6. Priority Classification Agent
   - Classifies action items into High, Medium, or Low Priority using Pydantic structured output.
   - Aligns classifications using task identifiers and fuzzy matching.
   - Provides an explanation for each classification.

7. Final Output Node
   - Combines the results.
   - Produces the structured meeting report.

## 6. Conditional Routing

The workflow uses conditional edges in LangGraph.

### When action items exist

Input Validation
    -> Topic Extraction
    -> Meeting Summary
    -> Action Item Extraction
    -> Priority Classification
    -> Final Report

### When no action items exist

Input Validation
    -> Topic Extraction
    -> Meeting Summary
    -> Action Item Extraction
    -> Final Report

The priority-classification agent is skipped when no action items
are found.

## 7. Priority Classification

### High Priority
Tasks explicitly marked urgent, critical, or subject to an imminent
deadline.

### Medium Priority
Important tasks with an upcoming deadline or tasks whose urgency
is not explicitly stated in the transcript.

### Low Priority
Routine or optional tasks without an urgent deadline.

The model uses the transcript as evidence. Tasks without explicit urgency
are defaulted to Medium Priority with an explanatory note.

## 8. Application Features

- User-provided meeting transcript input.
- Word and character counts.
- AI-generated meeting summary.
- Key discussion topics.
- Action items and task owners.
- Priority classification and explanations.
- Priority distribution metrics.
- JSON report download.
- CSV action-item download.
- Local JSON report saving.
- Error handling and troubleshooting messages.

## 9. Installation

Open Command Prompt or Terminal in the project folder.

Activate your existing virtual environment if necessary.

Install the required packages:

    python -m pip install streamlit pandas langgraph langchain-core langchain-ollama pydantic python-dotenv

Ensure Ollama is installed and running.

Check available models:

    ollama list

If you need to download the model used by default:

    ollama pull llama3.2

The first download requires an internet connection. Once the model
is available locally, inference runs completely offline.

## 10. Optional Environment Configuration

Create a `.env` file in the project folder if you want to configure
the model and Ollama connection:

    OLLAMA_MODEL=llama3.2
    OLLAMA_BASE_URL=http://localhost:11434
    TEMPERATURE=0

## 11. Running the Application

Open Command Prompt or Terminal in your project root directory.

Ensure Ollama is active (`ollama serve`), then launch Streamlit:

    python -m streamlit run app.py

Open the local URL printed in the terminal by Streamlit.

Paste a meeting transcript into the input area and click
**Analyze Meeting**.

## 12. Running the Workflow Separately

The workflow can also be tested directly from the command line:

    python workflow.py

Enter your transcript on the terminal, then type `END` on a separate
line to submit it.

The workflow prints the resulting JSON report.

## 13. Expected Output

The generated report contains:

- Meeting summary
- Key topics
- Action items
- Task owners
- Priority levels
- Priority explanations
- Priority classification status
- Total topics
- Total action items

The application also supports exporting the report as JSON and
exporting action items as CSV.

## 14. Limitations

- Output quality depends on the selected language model.
- Incomplete transcripts can produce incomplete results.
- Task ownership and deadlines cannot always be inferred reliably if omitted in the discussion.
- Local inference speed depends on your hardware and chosen model size.

## 15. Conclusion

The AI Meeting Notes Analyzer demonstrates how LangGraph coordinates
multiple AI processing steps into a structured, production-ready workflow.