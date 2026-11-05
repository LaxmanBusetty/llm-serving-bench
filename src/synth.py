"""Synthetic receipt generator with known gold labels.

Used for two things:
  1. A realistic, checkable workload for quality evaluation (field-level accuracy).
  2. A prompt set for load testing with a long shared prefix (to show prefix caching).

Everything is seeded, so the same seed always gives the same receipts.
"""
from __future__ import annotations

import json
import random
from datetime import date, timedelta

VENDORS = [
    "Greenfield Market", "Blue Harbor Cafe", "Sunrise Hardware", "Maple Street Bakery",
    "Northside Pharmacy", "Copper Kettle Diner", "Urban Pantry", "Lakeview Auto Parts",
    "Daily Brew Coffee", "Oak & Stone Supply", "Harvest Table Grocers", "Pixel Print Shop",
]

ITEMS = [
    ("Ice bags", 3.50), ("Paper towels", 7.25), ("Coffee beans 1lb", 12.99),
    ("Sourdough loaf", 6.50), ("Hand sanitizer", 4.75), ("Printer paper", 9.99),
    ("Batteries AA 8pk", 11.49), ("Olive oil", 14.20), ("Cleaning spray", 5.60),
    ("Napkins", 2.95), ("Takeout boxes", 18.00), ("Whole milk", 4.30),
    ("Zip ties", 6.15), ("Duct tape", 8.40), ("Sparkling water 12pk", 7.80),
]

PAYMENTS = [
    ("VISA ****{n}", "card"), ("MASTERCARD ****{n}", "card"), ("CASH", "cash"),
    ("Apple Pay", "card"), ("DEBIT ****{n}", "card"),
]

DATE_FORMATS = ["%m/%d/%Y", "%b %d, %Y", "%Y-%m-%d", "%d %b %Y"]


def make_receipt(rng: random.Random) -> dict:
    vendor = rng.choice(VENDORS)
    d = date(2026, 1, 1) + timedelta(days=rng.randint(0, 300))
    n_items = rng.randint(2, 6)
    lines, subtotal = [], 0.0
    for name, price in rng.sample(ITEMS, n_items):
        qty = rng.randint(1, 3)
        amt = round(qty * price, 2)
        subtotal += amt
        lines.append((qty, name, amt))
    subtotal = round(subtotal, 2)
    tax_rate = rng.choice([0.0, 0.0725, 0.0875, 0.095])
    tax = round(subtotal * tax_rate, 2)
    total = round(subtotal + tax, 2)
    pay_text, pay_gold = rng.choice(PAYMENTS)
    pay_text = pay_text.format(n=rng.randint(1000, 9999))

    body = [vendor.upper(), f"Store #{rng.randint(10, 99)}", d.strftime(rng.choice(DATE_FORMATS)), "-" * 28]
    for qty, name, amt in lines:
        body.append(f"{qty} x {name:<22} ${amt:>7.2f}")
    body += ["-" * 28, f"SUBTOTAL  ${subtotal:.2f}", f"TAX       ${tax:.2f}", f"TOTAL     ${total:.2f}", pay_text]
    gold = {
        "vendor": vendor,
        "date": d.isoformat(),
        "item_count": n_items,
        "tax": tax,
        "total": total,
        "payment_method": pay_gold,
    }
    return {"text": "\n".join(body), "gold": gold}


SYSTEM_PROMPT = """You are a bookkeeping assistant that extracts structured data from receipt text.

Return ONLY one JSON object with exactly these keys:
  vendor          string, the business name in normal capitalization
  date            string, ISO format YYYY-MM-DD
  item_count      integer, the number of distinct line items (not the sum of quantities)
  tax             number, the tax amount in dollars (0 if none)
  total           number, the final total in dollars
  payment_method  string, either "cash" or "card"

Rules:
- Do not include any text before or after the JSON.
- Dates may appear in several formats (MM/DD/YYYY, "Mar 14, 2026", "14 Mar 2026", YYYY-MM-DD). Always output YYYY-MM-DD.
- Treat VISA, MASTERCARD, DEBIT and Apple Pay as "card". Treat CASH as "cash".
- Use the TOTAL line, not the SUBTOTAL line.
- If tax is not shown, output 0.
- Numbers must be plain numbers without a dollar sign.

Example input:
GREENFIELD MARKET
Store #12
03/14/2026
----------------------------
2 x Whole milk              $   8.60
----------------------------
SUBTOTAL  $8.60
TAX       $0.00
TOTAL     $8.60
CASH

Example output:
{"vendor": "Greenfield Market", "date": "2026-03-14", "item_count": 1, "tax": 0, "total": 8.6, "payment_method": "cash"}
"""


def build_messages(receipt_text: str, no_think: bool = True) -> list[dict]:
    user = f"Receipt:\n{receipt_text}\n\nReturn the JSON object."
    if no_think:
        user += " /no_think"  # Qwen3 soft switch: skip the long reasoning trace
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def make_dataset(n: int, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    return [make_receipt(rng) for _ in range(n)]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Write the synthetic receipt eval set to a JSONL file")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/eval_set.jsonl")
    a = ap.parse_args()
    with open(a.out, "w") as f:
        for r in make_dataset(a.n, a.seed):
            f.write(json.dumps(r) + "\n")
    print(f"wrote {a.n} receipts to {a.out}")
