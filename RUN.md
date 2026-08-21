# How to run this project

Every command below is meant to be pasted into the **VS Code integrated terminal**
(`` Ctrl+` `` to open it). Windows PowerShell is the default on Windows; the
macOS/Linux equivalent is given wherever it differs.

> **Prefer not to install anything?** The portfolio site has a **Run this project**
> button that opens this repository in a free GitHub Codespace, installs the
> dependencies and runs the whole pipeline for you:
> <https://nikhil201716.github.io/nikhil-data-portfolio/pages/project.html?id=08>

---

## 1. Prerequisites

```powershell
python --version    # 3.11 or newer
git --version
```

### Optional — the local LLM stages

This project has stages that use a local model through [Ollama](https://ollama.com). They are **optional**: without it those stages are skipped or fall back to their deterministic control arm, and every other stage runs normally.

```powershell
winget install Ollama.Ollama
ollama pull qwen2.5:0.5b
ollama list
```

---

---

## 2. One-time setup

```powershell
git clone https://github.com/Nikhil201716/08-AutoClaim-Intelligence-Platform.git
cd 08-AutoClaim-Intelligence-Platform
```

Create and activate a virtual environment. This keeps the project's dependencies
from colliding with anything else on your machine.

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

<details>
<summary>If PowerShell refuses to run the activation script</summary>

Windows blocks unsigned scripts by default. Allow them for your own user account only:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
</details>

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then install the dependencies:

```powershell
pip install -r requirements.txt
```

> **Tip:** with the venv active, VS Code will offer to select it as the interpreter.
> Accept — otherwise the Run and Debug buttons use your global Python and you get
> `ModuleNotFoundError`.

---

## 3. Run it

### Everything, in one command

```powershell
python scripts/run_pipeline.py
```

That runs every stage below in order and is the normal way to use this repository.

### Or one stage at a time

Useful when you are changing a single stage and do not want to rebuild everything.

| # | Command | What it does |
|---|---|---|
| 1 | `python scripts/generate_claims_dataset.py` | Generate claims, render the PDFs and damage images |
| 2 | `python document_ai/ocr_extract.py` | OCR every claim PDF |
| 3 | `python document_ai/regex_extraction.py` | Extract fields by pattern matching against the template |
| 4 | `python document_ai/llm_field_extraction.py` | Extract the same fields with the local model |
| 5 | `python document_ai/evaluate_extraction.py` | Score both extractors against ground truth |
| 6 | `python vision/train_damage_classifier.py` | Train the damage classifier from scratch |
| 7 | `python vision/predict.py` | Classify every claim image |
| 8 | `python agents/claim_reconciliation_agent.py` | Run the cross-modal reconciliation check |
| 9 | `python scripts/evaluate_reconciliation.py` | Grade the flags and attribute each false positive |

Each stage produces the input the next one consumes, so run them in this order.


---

## 4. Explore the results

```powershell
streamlit run dashboard/streamlit_app.py
```

Opens on <http://localhost:8501>. VS Code will offer to forward the port and open it in your browser.

The pipeline writes everything it measures into `reports/`. Those files are the
source of every number quoted on the portfolio site — nothing is typed by hand.

```powershell
ls reports
```

---

## 5. What a correct run looks like

Expect regex extraction at 1.0000 and LLM at 0.9750 - and read Chapter 4 of the notebook before concluding anything from that gap. Vision accuracy lands near 83.9%.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError` | the virtual environment is not active | re-run the activate command from step 2 |
| `FileNotFoundError` on a data file | an earlier stage was skipped | run the stages in the documented order, or use the one-command runner |
| `Activate.ps1 cannot be loaded` | PowerShell execution policy | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| Numbers differ from the README | a seed or parameter changed | check the constants at the top of the generator script |
| `command not found` | dependency missing from this environment | `pip install -r requirements.txt` with the venv active |
| VS Code runs the wrong Python | interpreter not selected | `Ctrl+Shift+P` → *Python: Select Interpreter* → pick `.venv` |

---

## 7. Finish

```powershell
deactivate
```

---

## More

- **The 60+ page technical notebook** for this project is in [`docs/`](docs/) — it
  covers the business problem, the mathematics derived from first principles, a
  guided tour of the code, worked numerical examples and exercises with solutions.
- **All fifteen projects:** <https://nikhil201716.github.io/nikhil-data-portfolio/>

*Generated from this repository's own pipeline runner, so the stage list cannot
drift from the code.*
