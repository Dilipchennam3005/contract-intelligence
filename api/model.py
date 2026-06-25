"""
Model singleton — loaded once at startup, reused across all requests.
Swap MODEL_DIR to point at a better checkpoint without touching any API code.
"""
import os
from pathlib import Path

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

MODEL_DIR = Path(os.getenv("MODEL_DIR",
    str(Path(__file__).resolve().parents[1] / "models" / "final" / "step4-finetune-quick-cpu")
))
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN   = 256
STRIDE    = 64
MAX_WIN   = 10

# 41 CUAD clause types — the question template the model was trained on
CLAUSE_TYPES = [
    "Document Name", "Parties", "Agreement Date", "Effective Date",
    "Expiration Date", "Renewal Term", "Notice Period To Terminate Renewal",
    "Governing Law", "Most Favored Nation", "Non-Compete", "Exclusivity",
    "No-Solicit Of Customers", "Competitive Restriction Exception",
    "No-Solicit Of Employees", "Non-Disparagement", "Termination For Convenience",
    "Rofr/Rofo/Rofn", "Change Of Control", "Anti-Assignment",
    "Revenue/Profit Sharing", "Price Restrictions", "Minimum Commitment",
    "Volume Restriction", "Ip Ownership Assignment", "Joint Ip Ownership",
    "License Grant", "Non-Transferable License", "Affiliate License-Licensor",
    "Affiliate License-Licensee", "Unlimited/All-You-Can-Eat-License",
    "Irrevocable Or Perpetual License", "Source Code Escrow",
    "Post-Termination Services", "Audit Rights", "Uncapped Liability",
    "Cap On Liability", "Liquidated Damages", "Warranty Duration",
    "Insurance", "Covenant Not To Sue", "Third Party Beneficiary",
]

QUESTION_TEMPLATE = (
    'Highlight the parts (if any) of this contract related to "{clause}" '
    'that should be reviewed by a lawyer. Details: {description}'
)

CLAUSE_DESCRIPTIONS = {
    "Document Name": "The name of the contract",
    "Parties": "The two or more parties who signed the contract",
    "Agreement Date": "The date of the contract",
    "Effective Date": "The date when the contract is effective",
    "Expiration Date": "On what date will the contract's initial term expire?",
    "Renewal Term": "What is the renewal term after the initial term expires?",
    "Notice Period To Terminate Renewal": "Notice period to terminate renewal",
    "Governing Law": "Which state/country's law governs the contract?",
    "Most Favored Nation": "Most favored nation clause",
    "Non-Compete": "Is there a non-compete clause?",
    "Exclusivity": "Is there an exclusivity clause?",
    "No-Solicit Of Customers": "No-solicit of customers clause",
    "Competitive Restriction Exception": "Exceptions to competitive restrictions",
    "No-Solicit Of Employees": "No-solicit of employees clause",
    "Non-Disparagement": "Non-disparagement clause",
    "Termination For Convenience": "Can either party terminate for convenience?",
    "Rofr/Rofo/Rofn": "Right of first refusal, first offer, or first negotiation",
    "Change Of Control": "Change of control clause",
    "Anti-Assignment": "Is there an anti-assignment clause?",
    "Revenue/Profit Sharing": "Revenue or profit sharing arrangement",
    "Price Restrictions": "Price restriction clause",
    "Minimum Commitment": "Minimum commitment clause",
    "Volume Restriction": "Volume restriction clause",
    "Ip Ownership Assignment": "IP ownership assignment clause",
    "Joint Ip Ownership": "Joint IP ownership clause",
    "License Grant": "License grant clause",
    "Non-Transferable License": "Non-transferable license clause",
    "Affiliate License-Licensor": "Affiliate license for licensor",
    "Affiliate License-Licensee": "Affiliate license for licensee",
    "Unlimited/All-You-Can-Eat-License": "Unlimited or all-you-can-eat license",
    "Irrevocable Or Perpetual License": "Irrevocable or perpetual license",
    "Source Code Escrow": "Source code escrow clause",
    "Post-Termination Services": "Post-termination services clause",
    "Audit Rights": "Audit rights clause",
    "Uncapped Liability": "Uncapped liability clause",
    "Cap On Liability": "Cap on liability clause",
    "Liquidated Damages": "Liquidated damages clause",
    "Warranty Duration": "Warranty duration clause",
    "Insurance": "Insurance clause",
    "Covenant Not To Sue": "Covenant not to sue clause",
    "Third Party Beneficiary": "Third party beneficiary clause",
}


class ContractModel:
    def __init__(self):
        self.tokenizer = None
        self.model     = None
        self.loaded    = False

    def load(self):
        print(f"Loading model from {MODEL_DIR} on {DEVICE}...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        self.model     = AutoModelForQuestionAnswering.from_pretrained(str(MODEL_DIR)).to(DEVICE)
        self.model.eval()
        self.loaded    = True
        print("Model ready.")

    def predict_clause(self, context: str, clause_type: str) -> tuple[str, float]:
        """Returns (extracted_text, confidence_score). Empty string = not found."""
        desc     = CLAUSE_DESCRIPTIONS.get(clause_type, clause_type)
        question = QUESTION_TEMPLATE.format(clause=clause_type, description=desc)

        encoding = self.tokenizer(
            question, context,
            max_length=MAX_LEN, stride=STRIDE,
            truncation="only_second",
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
        )
        encoding.pop("offset_mapping")
        encoding.pop("overflow_to_sample_mapping", None)

        n_win = min(encoding["input_ids"].shape[0], MAX_WIN)
        best_score, best_text = float("-inf"), ""

        with torch.no_grad():
            for i in range(n_win):
                win = {k: v[i].unsqueeze(0).to(DEVICE) for k, v in encoding.items()}
                out = self.model(**win)
                s = out.start_logits[0].cpu()
                e = out.end_logits[0].cpu()

                start = int(s.argmax())
                end   = int(e.argmax())

                if start == 0 or end < start:
                    score = (s[0] + e[0]).item()
                    if score > best_score:
                        best_score, best_text = score, ""
                    continue

                score = (s[start] + e[end]).item()
                if score > best_score:
                    ids = encoding["input_ids"][i][start: end + 1]
                    best_text  = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
                    best_score = score

        # Normalise score to a rough 0-1 confidence
        confidence = float(torch.sigmoid(torch.tensor(best_score / 10)).item())
        return best_text, round(confidence, 3)

    def analyze(self, contract_text: str, title: str = "Untitled") -> dict:
        results = []
        for clause in CLAUSE_TYPES:
            text, conf = self.predict_clause(contract_text, clause)
            results.append({
                "clause_type":    clause,
                "found":          bool(text),
                "extracted_text": text if text else None,
                "confidence":     conf,
            })

        found = [r for r in results if r["found"]]
        return {
            "contract_title":        title,
            "total_clauses_checked": len(CLAUSE_TYPES),
            "clauses_found":         len(found),
            "results":               results,
            "model_version":         MODEL_DIR.name,
        }


# singleton — imported by FastAPI app
contract_model = ContractModel()
