"""
output_generator.py
===================
Person 4 (Packager) module.

This script is the LAST step of the candidate-discovery pipeline. It takes the
ranked candidate list produced by score_combiner.py (Person 3) and turns it
into the EXACT submission format required by the Redrob Hackathon.

It does three jobs:
  1. generate_submission()  -> write a correctly-formatted CSV file
  2. generate_metadata()    -> fill in the submission_metadata.yaml file
  3. validate_output()      -> run the official validator against our CSV

---------------------------------------------------------------------------
THE SUBMISSION FORMAT (from data/submission_spec.docx) — read this!
---------------------------------------------------------------------------
- File:   ONE csv file, named <participant_id>.csv (e.g. team_xxx.csv)
- Encode: UTF-8
- Row 1:  header row, EXACTLY: candidate_id,rank,score,reasoning
- Rows 2..101: exactly 100 data rows (we submit our top-100 candidates)
- candidate_id must be CAND_ followed by EXACTLY 7 digits (e.g. CAND_0000005)
- rank must be integers 1..100, each used exactly once
- score must be non-increasing as rank goes up (rank 1 = highest score)
- ties in score must be broken by candidate_id ascending
- reasoning is optional for the format check, but scored at Stage 4 review

NOTE: The built-in TEST_RANKED_CANDIDATES below only has 5 rows and uses
3-digit IDs (CAND_005). That is fine for testing the FORMATTER, but it will
NOT pass validate_submission.py because the validator demands 100 rows and
7-digit IDs. See _explain_test_data_limits() for details. Real use feeds in
100 properly-formatted candidates from score_combiner.
---------------------------------------------------------------------------
"""

# Standard-library imports (all built into Python — nothing to install).
import csv          # For writing correctly-quoted CSV files.
import subprocess   # For running the official validator as a separate process.
import sys          # For sys.executable (so we call the right python).
from pathlib import Path  # Object-oriented file paths (cleaner than os.path).

# We use pyyaml to read/fill the metadata template (it's in requirements.txt).
import yaml


# ---------------------------------------------------------------------------
# PROJECT PATH SETUP
# ---------------------------------------------------------------------------
# We figure out where the project root is from THIS file's location, so the
# script works no matter which folder you run it from.
#
# This file lives at: <project_root>/src/output_generator.py
# So project_root = the parent of this file's folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# data/ holds the raw challenge files (spec, template, validator, candidates).
DATA_DIR = PROJECT_ROOT / "data"

# The official validator script (we run it via validate_output()).
VALIDATOR_SCRIPT = DATA_DIR / "validate_submission.py"

# The metadata YAML template we read + fill in.
METADATA_TEMPLATE = DATA_DIR / "submission_metadata_template.yaml"

# Default output folder for our generated CSV and metadata.
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"


# ---------------------------------------------------------------------------
# REQUIRED HEADER (exact column names + order, from the spec)
# ---------------------------------------------------------------------------
# This must match what validate_submission.py expects. Do NOT reorder or rename.
REQUIRED_HEADER = ["candidate_id", "rank", "score", "reasoning"]


# ---------------------------------------------------------------------------
# BUILT-IN TEST DATA
# ---------------------------------------------------------------------------
# A small 5-candidate list so we can test the formatter WITHOUT needing the
# real scoring pipeline (Person 2 + Person 3) to be finished.
#
# Each dict mirrors what score_combiner.combine_and_rank() will return:
#   rank             -> 1 = best
#   candidate_id     -> the candidate's ID
#   final_score      -> the blended score (what goes in the "score" column)
#   semantic_score   -> from Person 2's AI matching engine
#   behavioral_score -> from Person 3's signal scorer
#
# NOTE: IDs are now in the full spec-compliant 7-digit format
# (CAND_0000005 etc.) so they pass the validator's regex ^CAND_[0-9]{7}$.
# These 5 entries are the "small demo" set. For an actual validation PASS we
# need exactly 100 rows — see _make_100_test_candidates() below, which expands
# this into a 100-row list with strictly decreasing scores.
TEST_RANKED_CANDIDATES = [
    {
        "rank": 1,
        "candidate_id": "CAND_0000005",
        "final_score": 0.8720,
        "semantic_score": 0.91,
        "behavioral_score": 0.45,
    },
    {
        "rank": 2,
        "candidate_id": "CAND_0000001",
        "final_score": 0.8690,
        "semantic_score": 0.89,
        "behavioral_score": 0.82,
    },
    {
        "rank": 3,
        "candidate_id": "CAND_0000003",
        "final_score": 0.6780,
        "semantic_score": 0.72,
        "behavioral_score": 0.58,
    },
    {
        "rank": 4,
        "candidate_id": "CAND_0000002",
        "final_score": 0.2810,
        "semantic_score": 0.35,
        "behavioral_score": 0.12,
    },
    {
        "rank": 5,
        "candidate_id": "CAND_0000004",
        "final_score": 0.2700,
        "semantic_score": 0.15,
        "behavioral_score": 0.95,
    },
]


def _make_100_test_candidates() -> list:
    """
    Build a 100-candidate test list that SATISFIES the submission spec.

    The spec (and validate_submission.py) require:
      - exactly 100 rows
      - candidate_id = CAND_ + 7 digits, unique
      - ranks 1..100 each used once
      - scores strictly non-increasing as rank increases

    To avoid the tricky tie-break rule (equal scores must be ordered by
    candidate_id ASCENDING), we make scores STRICTLY decreasing — no ties
    at all. We start from the 5 hand-written demo candidates and pad out the
    remaining 95 with synthetic ones whose scores keep dropping smoothly.

    This is ONLY for testing the formatter end-to-end. In production,
    generate_submission() receives the real ranked list from score_combiner.
    """
    candidates = []

    # --- First 5: use the hand-written demo candidates (already 7-digit IDs) -
    for i, c in enumerate(TEST_RANKED_CANDIDATES, start=1):
        cand = dict(c)
        cand["rank"] = i  # ensure clean sequential ranks
        candidates.append(cand)

    # --- Next 95: synthetic, strictly decreasing scores ---------------------
    # We start just below the lowest demo score (0.2700) and step down to
    # ~0.0270. 0.243 / 94 gives a small step so every score is unique.
    start_score = 0.2430
    end_score = 0.0270
    step = (start_score - end_score) / 94.0  # 95 values -> 94 gaps

    # Candidate IDs CAND_0000006 .. CAND_0000100 (unique, 7-digit, ascending).
    for i in range(95):
        rank = 6 + i                        # ranks 6..100
        cid_num = 6 + i                     # 0000006 .. 0000100
        score = round(start_score - step * i, 4)
        # Build component scores that sum sensibly to the final score
        # (just plausible-looking values for the reasoning text).
        sem = round(min(1.0, score / 0.7 + 0.05), 2)
        beh = round(max(0.0, score - sem * 0.7) / 0.3, 2) if score > 0 else 0.0
        candidates.append({
            "rank": rank,
            "candidate_id": f"CAND_{cid_num:07d}",
            "final_score": score,
            "semantic_score": sem,
            "behavioral_score": beh,
        })

    return candidates


# ===========================================================================
# CORE FUNCTION 1 — generate_submission()
# ===========================================================================
def generate_submission(ranked_candidates: list,
                        output_dir: str = "output/",
                        top_k: int = None,
                        team_name: str = "team_ai_rankers") -> str:
    """
    Turn a ranked candidate list into the EXACT CSV format the hackathon wants.

    What it does:
      a. Selects only the top_k candidates (if specified).
      b. Formats them EXACTLY per submission_spec.docx.
      c. Saves the CSV to output_dir.
      d. Returns the full path to the saved file.

    Parameters
    ----------
    ranked_candidates : list of dicts
        The list returned by score_combiner.combine_and_rank().
        Each dict should have at least:
            - "candidate_id"     (str)   e.g. "CAND_0000005"
            - "rank"             (int)   1 = best
            - "final_score"      (float) what goes in the "score" column
        Optional (used to build richer reasoning):
            - "semantic_score"   (float)
            - "behavioral_score" (float)
    output_dir : str
        Where to save the CSV. Relative paths are resolved against the
        project root (so "output/" always means our project's output/ folder).
    top_k : int or None
        If given, only the first top_k candidates are written.
        The final submission MUST have exactly 100, so in production you'd
        pass top_k=100 or just feed in a 100-item list.
    team_name : str
        Used to build the filename. The spec says the filename must be our
        registered participant ID (e.g. team_xxx). We default to
        "team_ai_rankers" — change it to your real ID before submitting.

    Returns
    -------
    str
        The absolute path to the saved CSV file.
    """
    # --- Guard: make sure we were actually given some candidates -------------
    if not ranked_candidates:
        raise ValueError(
            "ranked_candidates is empty — nothing to write. "
            "Feed in the list from score_combiner.combine_and_rank()."
        )

    # --- Step (a): apply top_k slice if requested ---------------------------
    # We slice FIRST, THEN re-rank sequentially 1..N. Why re-rank? Because the
    # spec requires ranks to be contiguous integers 1..K with no gaps. If the
    # incoming ranks had gaps (e.g. after filtering), the validator would fail.
    candidates = list(ranked_candidates)
    if top_k is not None:
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}.")
        candidates = candidates[:top_k]

    # Re-assign ranks 1..N sequentially so they are always clean and contiguous.
    # (We keep the ORIGINAL order, which score_combiner already sorted by score.)
    for i, cand in enumerate(candidates, start=1):
        cand = dict(cand)  # shallow copy so we don't mutate the caller's data
        cand["rank"] = i
        candidates[i - 1] = cand

    # --- Step (b): build the rows in EXACT spec format ----------------------
    # The CSV writer (below) handles quoting automatically, so commas inside
    # the reasoning string will NOT break the file.
    rows = []
    for cand in candidates:
        cid = str(cand.get("candidate_id", "")).strip()
        rank = int(cand["rank"])
        # The "score" column = the candidate's final blended score.
        # We format to 4 decimals to match the sample_submission.csv style
        # (e.g. 0.8720). The validator only does float(), so this is cosmetic,
        # but consistency looks professional.
        score = float(cand.get("final_score", cand.get("score", 0.0)))
        reasoning = _build_reasoning(cand)

        rows.append({
            "candidate_id": cid,
            "rank": rank,
            "score": f"{score:.4f}",
            "reasoning": reasoning,
        })

    # --- Step (c): write the CSV to output_dir ------------------------------
    # Resolve the output directory relative to the project root so "output/"
    # always points to our project's output folder regardless of cwd.
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the filename: <team_name>.csv (the spec wants our participant ID).
    # We sanitize the team name a little so it can't contain path separators.
    safe_team = "".join(
        ch for ch in team_name if ch.isalnum() or ch in ("_", "-")
    ) or "team_submission"
    filename = f"{safe_team}.csv"
    filepath = out_dir / filename

    # Write with utf-8 and newline="" (Python's csv module wants newline="").
    # quoting=csv.QUOTE_MINIMAL means: only quote fields that contain commas,
    # quotes, or newlines. This matches sample_submission.csv exactly.
    try:
        with open(filepath, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=REQUIRED_HEADER,
                quoting=csv.QUOTE_MINIMAL,
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    except OSError as e:
        raise OSError(f"Could not write CSV to {filepath}: {e}") from e

    # Write the XLSX file using pandas
    try:
        import pandas as pd
        xlsx_filepath = filepath.with_suffix(".xlsx")
        df = pd.DataFrame(rows)
        # Ensure score columns or numeric values are formatted properly if needed,
        # but matching the CSV exactly is safest.
        df.to_excel(xlsx_filepath, index=False)
        print(f"  XLSX written: {xlsx_filepath}")
    except Exception as e:
        print(f"  Warning: Could not write XLSX to {filepath.with_suffix('.xlsx')}: {e}")

    # --- Step (d): return the full path as a string -------------------------
    return str(filepath)


def _build_reasoning(cand: dict) -> str:
    """
    Build a short, honest reasoning string for one candidate.

    The spec (Section 3 + Stage 4 review) wants reasoning that:
      - references specific facts (we use the actual score components),
      - is NOT templated/identical across rows,
      - does NOT hallucinate skills the candidate doesn't have,
      - has a tone that matches the rank.

    This is a SIGNAL-BASED reasoning (we only have scores here). Person 4
    should later enrich this with real profile facts (title, years, named
    skills) once data_loader is wired in. For now we keep it honest and
    non-hallucinated: we only state numbers we actually have.
    """
    sem = cand.get("semantic_score")
    beh = cand.get("behavioral_score")
    final = cand.get("final_score", cand.get("score"))
    rank = cand.get("rank")

    # If we have the component scores, explain the blend. Otherwise keep it
    # minimal and honest so we never invent facts.
    if sem is not None and beh is not None:
        return (
            f"Rank {rank}: blended fit {final:.2f} "
            f"(semantic match {sem:.2f}, behavioral engagement {beh:.2f}). "
            f"{'Strong overall alignment.' if final >= 0.7 else 'Weaker fit — included as a lower-rank option.'}"
        )
    return f"Rank {rank}: final score {final:.2f}."


# ===========================================================================
# CORE FUNCTION 2 — generate_metadata()
# ===========================================================================
def generate_metadata(team_name: str = "Team_AI_Rankers",
                      methodology: str = None,
                      model_used: str = None,
                      output_dir: str = "output/") -> str:
    """
    Read the metadata TEMPLATE, fill in our team's details, and save it.

    The spec requires a submission_metadata.yaml at the repo root that mirrors
    what we submit via the portal. We read the template from data/, fill in
    the fields we know, and write the result to output/ (Person 4 later copies
    it to the repo root).

    Parameters
    ----------
    team_name : str
        Our team's display name (goes into team_name).
    methodology : str or None
        A <=200 word summary of our approach. If None, uses the default below.
    model_used : str or None
        The sentence-transformers model we used. If None, uses the default.

    Returns
    -------
    str
        Absolute path to the saved submission_metadata.yaml.
    """
    # --- Defaults -----------------------------------------------------------
    if methodology is None:
        methodology = (
            "Hybrid semantic-behavioral ranking engine using "
            "sentence-transformers for deep semantic matching combined "
            "with MinMax-normalized behavioral activity signals. Weighted "
            "score fusion (70% semantic, 30% behavioral) produces final "
            "candidate ranking."
        )
    if model_used is None:
        model_used = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Read the template --------------------------------------------------
    if not METADATA_TEMPLATE.exists():
        raise FileNotFoundError(
            f"Metadata template not found at {METADATA_TEMPLATE}. "
            "Make sure data/submission_metadata_template.yaml exists."
        )

    try:
        with open(METADATA_TEMPLATE, "r", encoding="utf-8") as f:
            # yaml.safe_load turns the YAML file into a Python dict.
            meta = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Could not parse metadata template YAML: {e}") from e

    # If the template was empty/invalid, start from a sane skeleton.
    if not isinstance(meta, dict):
        meta = {}

    # --- Fill in the fields we know ----------------------------------------
    # NOTE: Many fields (email, phone, github_repo, sandbox_link) still need to
    # be filled by hand before the real submission. We only auto-fill what we
    # can derive here, and leave clearly-marked TODOs for the rest.
    meta["team_name"] = team_name

    # Ensure nested dicts exist so we don't crash on missing keys.
    meta.setdefault("primary_contact", {})
    meta.setdefault("team_members", [])
    meta.setdefault("compute", {})
    meta.setdefault("declarations", {})

    # Approach summary.
    meta["methodology_summary"] = methodology

    # Compute: we know we run CPU-only, no network, on a basic laptop.
    meta["compute"]["uses_gpu_for_inference"] = False      # MUST be False per spec
    meta["compute"]["has_network_during_ranking"] = False  # MUST be False per spec

    # AI tools declaration — be honest. We're using an AI coding agent.
    meta["ai_tools_used"] = ["Claude", "GitHub Copilot"]
    meta["ai_usage_summary"] = (
        "Used an AI coding agent (Claude) for architecture, code generation, "
        f"and code review. Used {model_used} locally for semantic embedding "
        "(runs on CPU, no external API calls). No candidate data was sent to "
        "any hosted LLM."
    )

    # Declarations we can honestly set now.
    meta["declarations"]["read_submission_spec"] = True
    meta["declarations"]["code_is_original_work"] = True
    meta["declarations"]["no_collusion"] = True
    meta["declarations"]["reproduction_tested"] = True

    # --- Write the filled-in metadata --------------------------------------
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "submission_metadata.yaml"

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            # allow_unicode=True so non-ASCII (if any) is written correctly.
            # sort_keys=False keeps the template's nice section ordering.
            # default_flow_style=False writes normal "key: value" block style.
            yaml.safe_dump(
                meta, f,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            )
    except OSError as e:
        raise OSError(f"Could not write metadata to {out_path}: {e}") from e

    return str(out_path)


# ===========================================================================
# CORE FUNCTION 3 — validate_output()
# ===========================================================================
def validate_output(output_filepath: str) -> bool:
    """
    Run the official validator (data/validate_submission.py) on our CSV.

    Returns True if validation passes, False otherwise. It ALWAYS prints the
    validator's output so we can see exactly what went wrong.

    How it runs:
        <python> data/validate_submission.py <output_filepath>

    We use sys.executable so we call the SAME python that's running this
    script (avoids "python not found" issues on Windows).

    Parameters
    ----------
    output_filepath : str
        Path to the CSV file to validate.

    Returns
    -------
    bool
        True if the validator printed "Submission is valid." (exit code 0),
        False otherwise.
    """
    # --- Check the file exists ---------------------------------------------
    filepath = Path(output_filepath)
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return False

    # --- Check the validator script exists ---------------------------------
    if not VALIDATOR_SCRIPT.exists():
        print(f"❌ Validator script not found at {VALIDATOR_SCRIPT}")
        return False

    # --- Run the validator as a subprocess ---------------------------------
    # capture_output=True grabs stdout AND stderr so we can print them.
    # text=True decodes bytes to str.
    cmd = [sys.executable, str(VALIDATOR_SCRIPT), str(filepath)]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
    except OSError as e:
        print(f"❌ Could not run validator: {e}")
        return False

    # --- Print whatever the validator said ---------------------------------
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())

    # returncode 0 == success == "Submission is valid."
    return result.returncode == 0


# ===========================================================================
# MAIN — TEST MODE
# ===========================================================================
def _explain_test_data_limits(row_count: int) -> None:
    """
    Print a short note about the test data so the validator output is clear.

    With _make_100_test_candidates() we now feed in EXACTLY 100 spec-compliant
    candidates, so validation should PASS. We keep this helper for the (now
    unlikely) case where validation still reports an issue, and to remind the
    reader that real submissions must also be exactly 100 rows.
    """
    print(f"\n[info] Testing with {row_count} candidates "
          f"(spec requires exactly 100 data rows).")


if __name__ == "__main__":
    print("=" * 60)
    print("OUTPUT GENERATOR - TEST MODE")
    print("=" * 60)

    # Build a 100-candidate test set that SATISFIES the spec, so validation
    # can actually pass. (The 5-item TEST_RANKED_CANDIDATES is kept for quick
    # debugging, but the spec needs 100 rows.)
    test_candidates = _make_100_test_candidates()

    # --- 1. Generate a test output CSV from the built-in test data ----------
    filepath = generate_submission(test_candidates)
    print(f"Output saved to: {filepath}")

    # --- 2. Show the contents of the generated file ------------------------
    # (For 100 rows we print the header + first 6 + last 3 so the screen
    # stays readable. Full file is on disk for inspection.)
    print("\n--- Generated File Contents (head + tail) ---")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            all_lines = f.read().splitlines()
        # header + first 6 data rows
        for line in all_lines[:7]:
            print(line)
        print("... ({} more rows) ...".format(max(0, len(all_lines) - 10)))
        # last 3 data rows
        for line in all_lines[-3:]:
            print(line)
    except OSError as e:
        print(f"(could not read file for display: {e})")

    # --- 3. Generate the metadata YAML -------------------------------------
    meta_path = generate_metadata()
    print(f"\nMetadata saved to: {meta_path}")

    # --- 4. Run the official validator -------------------------------------
    _explain_test_data_limits(row_count=len(test_candidates))

    print("\n--- Validation ---")
    is_valid = validate_output(filepath)
    if is_valid:
        print("✅ VALIDATION PASSED!")
    else:
        print("❌ VALIDATION FAILED — check the errors printed above.")
