"""
promptlog - a tiny LLM logger.

Sends a prompt to an AI API (OpenAI or Anthropic), prints the answer, and
appends one row per call to a CSV file so you can see prompts, tokens,
latency and cost.

Usage:
    python promptlog.py "Explain recursion in one sentence"
    python promptlog.py --file prompt.txt --model gpt-5.4
    python promptlog.py --provider anthropic "Hello"
    python promptlog.py stats
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Which provider to use when --provider is not given.
# Change this to "anthropic" if you mostly use Claude.
DEFAULT_PROVIDER = "openai"

# Default model for each provider (used when --model is not given).
DEFAULT_MODELS = {
    "openai": "gpt-5.4-mini",
    "anthropic": "claude-opus-5-5",
}

# The environment variable that holds each provider's API key.
API_KEY_VARIABLES = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

# Where to get a key for each provider (shown when the key is missing).
API_KEY_PAGES = {
    "openai": "https://platform.openai.com/api-keys",
    "anthropic": "https://console.anthropic.com/",
}

DEFAULT_LOG = "promptlog.csv"
DEFAULT_MAX_TOKENS = 1024

# prices.json lives next to this script, so it is found from any folder.
PRICES_FILE = Path(__file__).parent / "prices.json"

CSV_COLUMNS = [
    "timestamp", "model", "prompt", "response",
    "input_tokens", "output_tokens", "total_tokens",
    "latency_ms", "cost_usd", "status", "error",
]


# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------

def parse_args(argv):
    """Read the command-line options."""
    parser = argparse.ArgumentParser(
        prog="promptlog",
        description="Send a prompt to an AI API and log the call to a CSV file. "
                    "Run 'python promptlog.py stats' for a summary.",
    )
    parser.add_argument("prompt", nargs="?", help="the prompt text (or 'stats')")
    parser.add_argument("--file", help="read the prompt from this text file")
    parser.add_argument("--provider", choices=list(DEFAULT_MODELS), default=DEFAULT_PROVIDER,
                        help=f"which AI company's API to use (default: {DEFAULT_PROVIDER})")
    parser.add_argument("--model", help="model name (default: " +
                        ", ".join(f"{m} for {p}" for p, m in DEFAULT_MODELS.items()) + ")")
    parser.add_argument("--system", help="optional system prompt")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help=f"max tokens in the reply (default: {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--log", default=DEFAULT_LOG, help=f"CSV log path (default: {DEFAULT_LOG})")
    return parser.parse_args(argv)


def get_prompt_text(args):
    """Return the prompt from --file or from the command line."""
    if args.file:
        try:
            return Path(args.file).read_text(encoding="utf-8").strip()
        except OSError as e:
            exit_with_message(f"Could not read prompt file '{args.file}': {e.strerror}")
    if args.prompt:
        return args.prompt
    exit_with_message('No prompt given. Example: python promptlog.py "Explain recursion in one sentence"')


def exit_with_message(message, code=1):
    """Print a friendly message to stderr and stop the program."""
    print(message, file=sys.stderr)
    sys.exit(code)


# ---------------------------------------------------------------------------
# Prices and cost
# ---------------------------------------------------------------------------

def load_prices():
    """Load per-model prices from prices.json (USD per million tokens)."""
    if not PRICES_FILE.exists():
        return {}
    try:
        data = json.loads(PRICES_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"Warning: prices.json is not valid JSON ({e}); costs will be empty.", file=sys.stderr)
        return {}
    # Ignore keys starting with "_" (used for comments).
    return {model: price for model, price in data.items() if not model.startswith("_")}


def estimate_cost(model_names, input_tokens, output_tokens, prices):
    """Return the estimated cost in USD, or None if no model name has a price.

    model_names is a list tried in order. OpenAI often answers with a dated
    name like "gpt-5.4-mini-2026-03-17", so we also try the name you asked for.
    """
    price = next((prices[name] for name in model_names if name in prices), None)
    if price is None:
        return None
    input_cost = input_tokens / 1_000_000 * price["input_per_million"]
    output_cost = output_tokens / 1_000_000 * price["output_per_million"]
    return input_cost + output_cost


# ---------------------------------------------------------------------------
# Calling the API
#
# Each provider has its own function. Both return the same simple dict:
#   {"model", "text", "input_tokens", "output_tokens", "status", "note"}
# and raise CallFailed with a friendly message when the call goes wrong.
# That way the rest of the program doesn't care which provider was used.
# ---------------------------------------------------------------------------

class CallFailed(Exception):
    """Raised when an API call fails. The message is already user-friendly."""


def call_openai(model, prompt, system, max_tokens):
    """Send one prompt to OpenAI (Responses API) and return a result dict."""
    import openai  # imported here so the other provider works without it

    client = openai.OpenAI()  # reads OPENAI_API_KEY from the environment
    request = {"model": model, "input": prompt, "max_output_tokens": max_tokens}
    if system:
        request["instructions"] = system

    try:
        response = client.responses.create(**request)
    except openai.AuthenticationError:
        raise CallFailed("Authentication failed: your OPENAI_API_KEY looks invalid.")
    except openai.PermissionDeniedError:
        raise CallFailed("Permission denied: your API key cannot use this model.")
    except openai.NotFoundError:
        raise CallFailed("Not found: check the model name (--model).")
    except openai.RateLimitError:
        raise CallFailed("Rate limit or quota hit: wait a moment, or check your OpenAI billing.")
    except openai.BadRequestError as e:
        raise CallFailed(f"Bad request: {e.message}")
    except openai.APIStatusError as e:
        raise CallFailed(f"API error {e.status_code}: {e.message}")
    except openai.APIConnectionError:
        raise CallFailed("Could not reach the OpenAI API. Check your internet connection.")

    status, note = "ok", ""
    if response.status == "incomplete":
        note = "Reply was cut off at --max-tokens."
    # A refusal comes back as a "refusal" part inside a message.
    for item in response.output:
        if item.type == "message" and any(part.type == "refusal" for part in item.content):
            status, note = "refused", "The model declined to answer this prompt."

    return {
        "model": response.model,
        "text": response.output_text.strip(),
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "status": status,
        "note": note,
    }


# Claude models that support Anthropic's server-side "fallbacks" option: if the
# model declines a request for safety reasons, the API retries it on another model.
MODELS_WITH_FALLBACKS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}


def call_anthropic(model, prompt, system, max_tokens):
    """Send one prompt to Claude (Anthropic API) and return a result dict."""
    import anthropic  # imported here so the other provider works without it

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    request = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        request["system"] = system

    try:
        if model in MODELS_WITH_FALLBACKS:
            message = client.beta.messages.create(
                **request,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        else:
            message = client.messages.create(**request)
    except anthropic.AuthenticationError:
        raise CallFailed("Authentication failed: your ANTHROPIC_API_KEY looks invalid.")
    except anthropic.PermissionDeniedError:
        raise CallFailed("Permission denied: your API key cannot use this model.")
    except anthropic.NotFoundError:
        raise CallFailed("Not found: check the model name (--model).")
    except anthropic.RateLimitError:
        raise CallFailed("Rate limit hit: wait a moment and try again.")
    except anthropic.BadRequestError as e:
        raise CallFailed(f"Bad request: {e.message}")
    except anthropic.APIStatusError as e:
        raise CallFailed(f"API error {e.status_code}: {e.message}")
    except anthropic.APIConnectionError:
        raise CallFailed("Could not reach the Anthropic API. Check your internet connection.")

    status, note = "ok", ""
    if message.stop_reason == "refusal":
        status, note = "refused", "The model declined to answer this prompt."
    elif message.stop_reason == "max_tokens":
        note = "Reply was cut off at --max-tokens."

    # Join the text blocks of the reply (skipping thinking blocks etc.).
    text = "".join(block.text for block in message.content if block.type == "text")

    return {
        # If a fallback happened, message.model is the model that actually answered.
        "model": message.model,
        "text": text.strip(),
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
        "status": status,
        "note": note,
    }


# Maps each --provider value to the function that calls it.
PROVIDER_FUNCTIONS = {
    "openai": call_openai,
    "anthropic": call_anthropic,
}


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------

def append_log_row(log_path, row):
    """Append one row to the CSV, writing the header first if the file is new."""
    path = Path(log_path)
    is_new_file = not path.exists() or path.stat().st_size == 0
    # utf-8-sig adds a marker so Excel shows non-English characters correctly.
    encoding = "utf-8-sig" if is_new_file else "utf-8"
    with path.open("a", newline="", encoding=encoding) as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if is_new_file:
            writer.writeheader()
        writer.writerow(row)


def make_row(model, prompt, response="", input_tokens=0, output_tokens=0,
             latency_ms=0, cost=None, status="ok", error=""):
    """Build a dict with every CSV column filled in."""
    return {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": model,
        "prompt": prompt,
        "response": response,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "latency_ms": latency_ms,
        "cost_usd": "" if cost is None else f"{cost:.6f}",
        "status": status,
        "error": error,
    }


# ---------------------------------------------------------------------------
# The "send a prompt" command
# ---------------------------------------------------------------------------

def run_prompt(args):
    """Send the prompt, print the reply, and log the call. Returns an exit code."""
    prompt = get_prompt_text(args)
    model = args.model or DEFAULT_MODELS[args.provider]

    key_variable = API_KEY_VARIABLES[args.provider]
    if not os.environ.get(key_variable):
        exit_with_message(
            f"{key_variable} is not set.\n"
            f"Get a key at {API_KEY_PAGES[args.provider]} and then set it:\n"
            f"  macOS/Linux:  export {key_variable}=\"your-key-here\"\n"
            f"  Windows (PowerShell):  $env:{key_variable}=\"your-key-here\"\n"
            f"  Windows (cmd):  set {key_variable}=your-key-here"
        )

    prices = load_prices()
    call_provider = PROVIDER_FUNCTIONS[args.provider]

    start = time.perf_counter()
    try:
        result = call_provider(model, prompt, args.system, args.max_tokens)
    except CallFailed as e:
        latency_ms = round((time.perf_counter() - start) * 1000)
        append_log_row(args.log, make_row(model, prompt, latency_ms=latency_ms,
                                          status="error", error=str(e)))
        print(f"Error: {e}", file=sys.stderr)
        print(f"(The failed call was logged to {args.log}.)", file=sys.stderr)
        return 1
    latency_ms = round((time.perf_counter() - start) * 1000)

    cost = estimate_cost([result["model"], model], result["input_tokens"], result["output_tokens"], prices)
    if cost is None:
        print(f"Warning: no price for '{model}' in prices.json; cost left empty.", file=sys.stderr)

    append_log_row(args.log, make_row(result["model"], prompt, result["text"],
                                      result["input_tokens"], result["output_tokens"],
                                      latency_ms, cost, result["status"], result["note"]))

    print(result["text"] or "(empty response)")
    if result["note"]:
        print(f"Note: {result['note']}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# The "stats" command
# ---------------------------------------------------------------------------

def to_number(value):
    """Convert a CSV cell to a float; empty or bad cells become 0."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def truncate(text, length=60):
    """Shorten text to `length` characters, on one line."""
    text = " ".join(text.split())
    return text if len(text) <= length else text[: length - 3] + "..."


def run_stats(log_path):
    """Print a short summary of the CSV log. Returns an exit code."""
    path = Path(log_path)
    if not path.exists():
        exit_with_message(f"No log file found at '{log_path}'. Send a prompt first.")

    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("The log is empty.")
        return 0

    total_calls = len(rows)
    errors = sum(1 for r in rows if r["status"] == "error")
    total_tokens = sum(int(to_number(r["total_tokens"])) for r in rows)
    total_cost = sum(to_number(r["cost_usd"]) for r in rows)
    unpriced = sum(1 for r in rows if r["status"] != "error" and r["cost_usd"] == "")
    successful = [r for r in rows if r["status"] != "error"]
    avg_latency = (sum(to_number(r["latency_ms"]) for r in successful) / len(successful)
                   if successful else 0)

    print(f"Log file:         {log_path}")
    print(f"Total calls:      {total_calls}")
    print(f"Errors:           {errors}")
    print(f"Total tokens:     {total_tokens:,}")
    print(f"Estimated cost:   ${total_cost:.4f}" + (f"  ({unpriced} call(s) without a price)" if unpriced else ""))
    print(f"Average latency:  {avg_latency:,.0f} ms (successful calls only)")

    priced = [r for r in rows if r["cost_usd"] != ""]
    top3 = sorted(priced, key=lambda r: to_number(r["cost_usd"]), reverse=True)[:3]
    if top3:
        print("\nMost expensive prompts:")
        for i, r in enumerate(top3, start=1):
            print(f"  {i}. ${to_number(r['cost_usd']):.6f}  {truncate(r['prompt'])}")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.prompt == "stats" and not args.file:
        return run_stats(args.log)
    return run_prompt(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        exit_with_message("\nCancelled.", code=130)
