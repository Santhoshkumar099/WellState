import base64
import json
import re
import csv
import os
from pathlib import Path
from datetime import datetime

import anthropic
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL")
CSV_OUTPUT_PATH = "extractions.csv"

client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

app = FastAPI(
    title="WellStat Utility Bill Extractor",
    description="Extract structured data from utility bill PDFs using Claude Vision.",
    version="1.0.0"
)

# ── Extraction prompt ─────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """
You are a highly precise utility bill data extraction engine for WellStat, an energy management platform.

Your task is to extract utility bill details from the attached PDF and return ONLY one valid JSON object matching the exact schema below.

STRICT RULES:
1. Return ONLY raw JSON — no markdown, no ```json fences, no explanations.
2. Do NOT add or remove keys.
3. If a value is missing or not present, set it to null.
4. Do NOT invent or guess values.
5. Monetary values must be numbers (no currency symbols, no commas).
6. All kWh, kW, and quantity values must be numbers.
7. Dates must be returned in YYYY-MM-DD format.
8. The output must be a single valid JSON object.

FIELD RULES:

- "utility_type": one of "electricity", "gas", "water", or "multi" (if more than one commodity on bill).
- "bill_id": the invoice or bill number — must always be present if visible.
- "total_demand": maximum demand in kW, null if not present.
- "total_consumption": value with unit, null if not present.
- "total_consumption_on_peak": on-peak value with unit only, null if not broken out.
- "total_consumption_off_peak": off-peak value with unit only, null if not broken out.
- "total_supply": total electricity supplied if separately stated, else null.
- "total_cost": same as total_bill_amount unless a subtotal excluding taxes is explicitly labeled.
- "total_on_peak_cost": dollar amount for on-peak energy charges only, null if not broken out.
- "total_off_peak_cost": dollar amount for off-peak energy charges only, null if not broken out.
- "outstanding_balance": any unpaid amount carried forward from prior periods.
- "new_charges": charges added this billing period only.
- "total_bill_amount": the final total amount due (all charges combined).
- "meter_read_type": "Actual" or "Estimated" based on how readings are labeled.
- "submeters": array of meter objects. Each entry has:
    - "submeter_number": meter serial/ID
    - "consumption": value with unit for that meter
    - "demand": value with unit for that meter, null if not listed
  If no submeters, return an empty array [].
- "overall_confidence_score": float between 0.0 and 1.0.

Calculate "overall_confidence_score" using field weights. Score 1.0 if reliable, 0.5 if partial, 0.0 if null and expected.

- "justification": 3 to 4 lines justification for confidence score.

JSON SCHEMA:
{
  "account_number": null,
  "bill_id": null,
  "utility_type": null,
  "supplier": null,
  "start_date": null,
  "end_date": null,
  "meter_read_type": null,
  "total_consumption": null,
  "total_consumption_on_peak": null,
  "total_consumption_off_peak": null,
  "total_demand": null,
  "total_supply": null,
  "total_cost": null,
  "total_on_peak_cost": null,
  "total_off_peak_cost": null,
  "outstanding_balance": null,
  "new_charges": null,
  "total_bill_amount": null,
  "overall_confidence_score": null,
  "justification": null,
  "submeters": []
}
"""

# ── Core extraction function ──────────────────────────────────────────────────

async def extract_pdf_with_claude(filename: str, pdf_bytes: bytes) -> dict:
    pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")

    response = await client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=8192,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_b64
                        }
                    },
                    {
                        "type": "text",
                        "text": EXTRACTION_PROMPT
                    }
                ]
            }
        ]
    )

    raw_text = "".join(
        block.text for block in response.content if hasattr(block, "text")
    )

    clean = re.sub(r"```(?:json)?\s*", "", raw_text).strip()
    json_match = re.search(r'\{.*\}', clean, re.DOTALL)

    if not json_match:
        raise ValueError(f"Claude did not return recognisable JSON:\n{raw_text}")

    try:
        parsed = json.loads(json_match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parse error: {e}\n\nRaw output:\n{json_match.group(0)}")

    parsed["_meta"] = {
        "source_file": filename,
        "model": CLAUDE_MODEL,
        "engine": "vision-parser",
        "status": "extracted"
    }

    return parsed

# ── CSV helpers ───────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    "extracted_at", "source_file",
    "account_number", "bill_id", "utility_type", "supplier",
    "start_date", "end_date", "meter_read_type",
    "total_consumption", "total_consumption_on_peak", "total_consumption_off_peak",
    "total_demand", "total_supply",
    "total_cost", "total_on_peak_cost", "total_off_peak_cost",
    "outstanding_balance", "new_charges", "total_bill_amount",
    "overall_confidence_score", "justification",
    "submeters_json"
]

def append_to_csv(data: dict, filename: str):
    file_exists = Path(CSV_OUTPUT_PATH).exists()

    row = {col: None for col in CSV_COLUMNS}
    row["extracted_at"] = datetime.utcnow().isoformat()
    row["source_file"] = filename

    for col in CSV_COLUMNS:
        if col in data:
            val = data[col]
            if isinstance(val, (dict, list)):
                row[col] = json.dumps(val)
            else:
                row[col] = val

    row["submeters_json"] = json.dumps(data.get("submeters", []))

    with open(CSV_OUTPUT_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post(
    "/extract",
    summary="Extract bill data as JSON",
    description="Upload a utility bill PDF and receive the extracted structured data as JSON.",
    response_class=JSONResponse
)
async def extract_to_json(file: UploadFile = File(..., description="Utility bill PDF")):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = await extract_pdf_with_claude(file.filename, pdf_bytes)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

    return JSONResponse(content=result)


@app.post(
    "/extract-to-csv",
    summary="Extract bill data and save to CSV",
    description="Upload a utility bill PDF. Extracted data is appended to 'extractions.csv' and returned as JSON.",
    response_class=JSONResponse
)
async def extract_to_csv(file: UploadFile = File(..., description="Utility bill PDF")):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = await extract_pdf_with_claude(file.filename, pdf_bytes)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

    try:
        append_to_csv(result, file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV write failed: {str(e)}")

    return JSONResponse(content={
        "message": f"Extraction complete. Row appended to '{CSV_OUTPUT_PATH}'.",
        "csv_path": CSV_OUTPUT_PATH,
        "data": result
    })


@app.get("/health", summary="Health check")
async def health():
    return {"status": "ok", "model": CLAUDE_MODEL}