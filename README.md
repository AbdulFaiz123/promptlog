# promptlog

A tiny LLM logger. It sends a prompt to an AI API (OpenAI by default, or Anthropic's Claude), prints the reply, and appends one row per call to a CSV file (`promptlog.csv`). You can open that file in Excel or Google Sheets to see what you sent, what came back, how long it took and what it cost.

## Setup

```bash
cd promptlog
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Set your API key

Get an OpenAI key from https://platform.openai.com/api-keys (your account needs billing credits). Never put it in the code.

macOS / Linux:
```bash
export OPENAI_API_KEY="sk-..."
```

Windows (PowerShell):
```powershell
$env:OPENAI_API_KEY="sk-..."
```

Windows (cmd):
```cmd
set OPENAI_API_KEY=sk-...
```

This only lasts for the current terminal window. To keep it, add the line to `~/.zshrc` / `~/.bashrc` (macOS/Linux) or use `setx OPENAI_API_KEY "sk-..."` (Windows, then open a new window).

To use Claude instead, set `ANTHROPIC_API_KEY` (key from https://console.anthropic.com/) the same way and add `--provider anthropic` to your commands.

## Examples

```bash
python promptlog.py "Explain recursion in one sentence"
python promptlog.py --file prompt.txt --system "Answer like a pirate" --max-tokens 300
python promptlog.py stats
```

Other options: `--provider anthropic` to use Claude, `--model gpt-5.4` to pick a model (default: `gpt-5.4-mini` for OpenAI, `claude-opus-5-5` for Anthropic), `--log mylog.csv` to use a different CSV file (also works with `stats`).

## Prices

`prices.json` holds the price per million tokens for each model. It ships with entries for the two default models (`gpt-5.4-mini` and `claude-opus-5-5`). Prices change, so check the provider's pricing page and edit the file. If you use a model that isn't listed, the cost column is left empty and you get a one-line warning.

## CSV columns

| Column | Meaning |
|---|---|
| `timestamp` | When the call was made, local time, ISO 8601 (e.g. `2026-10-19T23:40:12+02:00`) |
| `model` | The model that answered (OpenAI often adds a date, e.g. `gpt-5.4-mini-2026-03-17`) |
| `prompt` | The text you sent |
| `response` | The text that came back (empty on errors) |
| `input_tokens` | Tokens in your prompt (plus system prompt) |
| `output_tokens` | Tokens the model generated (includes its internal reasoning/thinking) |
| `total_tokens` | `input_tokens + output_tokens` |
| `latency_ms` | Milliseconds from sending the request to receiving the full reply |
| `cost_usd` | Estimated cost from `prices.json` (empty if the model has no price) |
| `status` | `ok`, `error` (call failed) or `refused` (model declined) |
| `error` | Error message, or a note such as "Reply was cut off at --max-tokens" |

## Note on Claude refusals

When using `--provider anthropic`, for models that support it (like the default `claude-opus-5-5`), promptlog turns on Anthropic's server-side **fallback**: if the model declines a prompt for safety reasons, the API automatically retries it on another model. The `model` column then shows which model actually answered.
