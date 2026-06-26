"""
semantic_scorer.py
==================
Person 2 (AI matching engine) module.

This script is the SEMANTIC brain of the candidate-discovery pipeline. It reads
a job description and a pile of candidate profiles, and asks:

    "Which candidates TALK about the same things the job asks for?"

It answers that question the modern way — with embeddings. We feed the job
description and each candidate's profile text through a sentence-transformers
model, which turns each block of text into a vector that captures its MEANING
(not just keyword overlap). Then we compare the job vector against every
candidate vector with cosine similarity, producing a score between 0.0
(totally unrelated) and 1.0 (nearly identical meaning).

---------------------------------------------------------------------------
WHY SEMANTIC MATCHING (and not just keyword matching)?
---------------------------------------------------------------------------
A job asking for "ML engineer with NLP experience" should match a candidate
who wrote "built language models for text classification" — even though they
share almost no words in common. Keyword matchers miss this; embeddings catch
it because they learn that "NLP" and "language models" live in similar
semantic space.

The score this module produces becomes the SEMANTIC component of the final
blended rank (see score_combiner.py, which fuses it with behavioral signals).

---------------------------------------------------------------------------
TWO SUPPORTED MODELS (you can switch at call time)
---------------------------------------------------------------------------
  - "sentence-transformers/all-MiniLM-L6-v2"   fast,  ~80MB, 384-dim
  - "BAAI/bge-base-en-v1.5"                    slower, ~420MB, 768-dim, more accurate

Pass `model_name=...` to compute_semantic_scores() to pick one. If you omit
it, MODEL_NAME (the fast default) is used. The test mode at the bottom runs
BOTH and prints a side-by-side comparison so you can decide which to submit.

---------------------------------------------------------------------------
HOW TO USE
---------------------------------------------------------------------------
    from semantic_scorer import compute_semantic_scores

    # Default (fast) model:
    scores = compute_semantic_scores(
        job_description="We want a Python ML engineer...",
        candidate_profiles={"CAND_001": "5 years of PyTorch...", ...},
    )

    # Or the more accurate model:
    scores = compute_semantic_scores(
        job_description="...",
        candidate_profiles={...},
        model_name="BAAI/bge-base-en-v1.5",
    )

Or just run this file directly to test BOTH models on the built-in data:

    python semantic_scorer.py

---------------------------------------------------------------------------
DEPENDENCIES (all in requirements.txt)
---------------------------------------------------------------------------
  - sentence-transformers   -> the embedding models
  - scikit-learn            -> cosine_similarity helper (numerically stable)
  - torch                   -> pulled in by sentence-transformers as its backend
  - numpy                   -> array reshaping for the similarity math
"""

# ===========================================================================
# IMPORTS
# ===========================================================================
# Standard-library first (built into Python — nothing to install).
import json  # For loading the real sample_5_candidates.json.
import sys   # For sys.exit on fatal errors during the test run.
import time  # For timing each model in the comparison test.
from pathlib import Path  # Object-oriented file paths (cleaner than os.path).

# Third-party libraries. These are heavy imports (torch + transformers load a
# LOT of code on import), so we do them at module level. That's fine here
# because anyone importing this module wants the embeddings anyway.
import numpy as np  # noqa: E402  (numpy import before sklearn usage below)


# ===========================================================================
# PROJECT PATH SETUP
# ===========================================================================
# We locate the project root from THIS file's location so loaders work no
# matter which folder you run the script from.
#
# This file lives at: <project_root>/src/semantic_scorer.py
# So project_root = the parent of this file's folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# data/ holds the real challenge files (sample candidates, job description).
DATA_DIR = PROJECT_ROOT / "data"

# The teammate-provided 5-candidate real sample (5 entries from the real
# candidates.jsonl, for testing the scorer on REAL profile shapes).
REAL_SAMPLE_FILE = DATA_DIR / "sample_5_candidates.json"

# The real job description (a .docx — we extract its text with python-docx).
JOB_DESCRIPTION_DOCX = DATA_DIR / "job_description.docx"


# ===========================================================================
# CONFIGURATION
# ===========================================================================
# The DEFAULT model used when the caller doesn't specify one. Picked for speed
# and small size — good enough for iteration, and a safe default.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Registry of the models this module knows how to drive. Adding a new model
# here is the ONE place you change to extend support: the rest of the module
# (dimensions lookup, validation, recommendation) reads from this dict.
#
# Fields:
#   vector_dimensions -> output size of the embedding (used for reshaping and
#                        for get_model_info(); must match the real model).
#   size_mb           -> approximate download size (for warning the user).
#   speed             -> human label, used in the recommendation blurb.
#   accuracy          -> human label, used in the recommendation blurb.
#   note              -> one-line description for prints.
SUPPORTED_MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2": {
        "vector_dimensions": 384,
        "size_mb": 80,
        "speed": "fast",
        "accuracy": "good",
        "note": "Fast and good enough for most use cases.",
    },
    "BAAI/bge-base-en-v1.5": {
        "vector_dimensions": 768,
        "size_mb": 420,
        "speed": "slow",
        "accuracy": "high",
        "note": "Slower download/run, but more accurate rankings.",
    },
}


# ===========================================================================
# BUILT-IN TEST DATA
# ===========================================================================
# A self-contained job description + 5 candidate profiles so this module can
# be tested STANDALONE (no other files needed). Just run:
#     python semantic_scorer.py
#
# The candidates are deliberately mixed so the test produces an interesting
# ranking:
#   CAND_001 -> Senior ML engineer      (should score HIGHEST)
#   CAND_005 -> AI research engineer    (should also score HIGH)
#   CAND_003 -> Data scientist          (MIDDLE — some overlap)
#   CAND_002 -> Full-stack web dev      (LOW — wrong domain)
#   CAND_004 -> Marketing manager       (LOWEST — totally unrelated)
#
# Note: these short profiles use 3-digit IDs (CAND_005) for readability in the
# test output. The real pipeline feeds in 7-digit spec-compliant IDs from
# data_loader; the scoring math is identical either way.
TEST_JOB_DESCRIPTION = """
We are looking for a Senior Machine Learning Engineer with 3+ years
of experience in Python, deep learning frameworks (PyTorch/TensorFlow),
natural language processing (NLP), and experience deploying ML models
to production. Experience with recommendation systems, search relevance,
and large language models is a plus. Must have strong communication
skills and ability to work in cross-functional teams.
"""

TEST_CANDIDATES = {
    "CAND_001": "Senior ML Engineer with 5 years of experience in Python, "
                "PyTorch, and TensorFlow. Built NLP models for text classification "
                "and sentiment analysis. Deployed models to AWS SageMaker. "
                "Led a team of 3 data scientists.",
    "CAND_002": "Full-stack web developer specializing in React.js and Node.js. "
                "3 years of experience building responsive web applications. "
                "Proficient in MongoDB, Express, and REST APIs.",
    "CAND_003": "Data Scientist with expertise in Python, scikit-learn, and pandas. "
                "2 years working on predictive models and statistical analysis. "
                "Experience with NLP and text mining projects.",
    "CAND_004": "Marketing manager with 7 years of experience in digital campaigns. "
                "Expert in SEO, content strategy, and social media marketing. "
                "MBA from a top business school.",
    "CAND_005": "AI Research Engineer specializing in large language models and "
                "recommendation systems. Published 3 papers on transformer architectures. "
                "4 years of Python and PyTorch experience.",
}


# ===========================================================================
# MODEL CACHE (so we don't reload a model on every call)
# ===========================================================================
# Loading a model is SLOW (a few seconds the first time, as it reads weights
# from disk and warms up). In a real pipeline run we call compute_semantic_scores
# ONCE, but the test mode runs TWO different models, and other code may call in
# too. We don't want to pay the load cost twice for the same model.
#
# So we keep a DICT of loaded models, keyed by model name. _get_model() fills
# the entry for a given name on first use and returns the cached object
# thereafter. Switching models loads the new one once, then it's cached too.
_MODEL_CACHE = {}


def _get_model(model_name: str):
    """
    Load (and cache) the sentence-transformers model named `model_name`.

    Returns the loaded model object. On any failure, raises a RuntimeError
    with a friendly hint about what to check (usually: forgot to run
    `pip install -r requirements.txt`, asked for an unsupported model, or no
    network for the first download).

    We keep this separate from compute_semantic_scores() so the caching is
    explicit and testable on its own.
    """
    # --- Validate the model name against our whitelist ---------------------
    # Catch typos early with a clear message instead of a deep HF error later.
    if model_name not in SUPPORTED_MODELS:
        supported = "\n  - ".join(SUPPORTED_MODELS.keys())
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported models are:\n"
            f"  - {supported}"
        )

    # --- Already loaded for this name? Return immediately ------------------
    # This is the whole point of the dict cache: each model pays its load cost
    # exactly once per process.
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    # --- Import here (not at top of file) so that: -------------------------
    #   1. `import semantic_scorer` doesn't drag in torch+transformers until
    #      we actually need them (faster import, friendlier errors if the
    #      heavy deps aren't installed).
    #   2. The error message below can mention sentence_transformers by name.
    try:
        # The library is imported as `sentence_transformers` (underscore),
        # even though the pip package is `sentence-transformers` (hyphen).
        from sentence_transformers import SentenceTransformer
        import torch
    except ImportError as e:
        # Most common cause: requirements.txt wasn't installed. Give a hint.
        raise ImportError(
            "Could not import sentence_transformers or torch. "
            "Install the dependencies first:\n"
            "    pip install -r requirements.txt\n"
            f"(original error: {e})"
        ) from e

    info = SUPPORTED_MODELS[model_name]
    print(f"Loading model... ({model_name})")
    print(f"    [{info['speed']} / {info['accuracy']} accuracy / "
          f"~{info['size_mb']}MB] {info['note']}")
    print("    (first run downloads from Hugging Face — this can take a "
          "minute or two; later runs use the local cache)")

    try:
        # SentenceTransformer(...) downloads the model the first time and
        # loads it from the local HF cache thereafter. device="cuda" is preferred if GPU is available.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name, device=device)
    except Exception as e:
        # Blanket except is intentional — model loading can fail many ways
        # (network, disk, corrupted cache, OOM). We just need to surface a
        # clear message rather than a deep stack trace. Make sure NOT to
        # cache a half-loaded failure.
        _MODEL_CACHE.pop(model_name, None)
        raise RuntimeError(
            f"Failed to load model '{model_name}'.\n"
            "Common fixes:\n"
            "  - Check your internet connection (first download needs it).\n"
            "  - Clear the HF cache: rm -rf ~/.cache/huggingface\n"
            "  - Verify the model name is spelled correctly.\n"
            f"(original error: {e})"
        ) from e

    print("    Model loaded successfully.")
    return _MODEL_CACHE[model_name]


# ===========================================================================
# CORE FUNCTION — compute_semantic_scores()
# ===========================================================================
def compute_semantic_scores(job_description: str,
                            candidate_profiles: dict,
                            model_name: str = None) -> dict:
    """
    Score how well each candidate's profile matches the job description,
    using sentence-embedding cosine similarity.

    This is the function other modules (score_combiner.py) call. It is the
    ONLY public entry point of this module.

    Parameters
    ----------
    job_description : str
        The full job-description text. Treated as a single "sentence" (the
        model handles multi-sentence paragraphs fine).
    candidate_profiles : dict
        Mapping of {candidate_id: profile_text}. Order doesn't matter — we
        preserve the keys in the returned dict. profile_text can be any
        length, but a few hundred words is the sweet spot for these models.
    model_name : str or None
        Which model to use. If None, falls back to MODEL_NAME (the fast
        default). Must be one of SUPPORTED_MODELS:
          - "sentence-transformers/all-MiniLM-L6-v2" (fast, good enough)
          - "BAAI/bge-base-en-v1.5"                  (slower, more accurate)

    Returns
    -------
    dict
        {candidate_id: similarity_score}, where similarity_score is a float
        between 0.0 (no semantic overlap) and 1.0 (near-identical meaning).
        Same keys as the input dict.

    Steps (each printed so you can watch progress):
        a. Resolve the model name (None -> MODEL_NAME) and load it (cached).
        b. Encode the job description into a single vector.
        c. Encode ALL candidate profiles into vectors IN ONE BATCH
           (encoding them all at once is much faster than a per-candidate loop,
           because the model can parallelize across the batch).
        d. Compute cosine similarity between the JD vector and each candidate
           vector using sklearn's numerically-stable cosine_similarity.
        e. Build and return the {candidate_id: score} dict.
    """
    # --- Resolve which model to use ----------------------------------------
    # None means "use the default". We resolve once here so the rest of the
    # function and any error messages refer to a concrete name.
    if model_name is None:
        model_name = MODEL_NAME

    # --- Input validation --------------------------------------------------
    # Fail fast with a clear message instead of crashing deep in numpy later.
    if not job_description or not job_description.strip():
        raise ValueError(
            "job_description is empty — semantic scoring needs actual text. "
            "Did data_loader fail to read the job description?"
        )
    if not candidate_profiles:
        raise ValueError(
            "candidate_profiles is empty — nothing to score. "
            "Did data_loader return an empty candidate set?"
        )

    # Defensive copy of the keys so the returned dict has a stable order even
    # if the caller mutates their input dict while we work.
    candidate_ids = list(candidate_profiles.keys())
    # Pull the profile texts out in the SAME ORDER as the ids. Keeping id and
    # text aligned by position is critical — the score at row i must map back
    # to candidate_ids[i].
    profile_texts = [candidate_profiles[cid] for cid in candidate_ids]

    # --- TEXT SANITISATION (prevents [Errno 22] Invalid argument) ----------
    # Some candidate profiles contain null bytes (\x00), surrogate code
    # points, or other characters that the tokeniser's underlying file I/O
    # or C extension cannot handle on Windows.  We strip them here.
    def _sanitise(text: str) -> str:
        if not isinstance(text, str):
            return ""
        # Remove null bytes and other C0 control chars except common whitespace
        return text.replace("\x00", "").encode("utf-8", errors="ignore").decode("utf-8")

    profile_texts = [_sanitise(t) for t in profile_texts]

    # Sanity-check: every profile must be a non-empty string. We skip empties
    # in scoring (assign them 0.0) rather than crashing the whole batch.
    for cid, text in zip(candidate_ids, profile_texts):
        if not isinstance(text, str) or not text.strip():
            print(f"⚠️  Warning: candidate {cid} has an empty/invalid profile; "
                  f"it will score 0.0.")

    # --- Step (a): load the model (cached per model_name) ------------------
    # _get_model() returns the cached model after the first call for a given
    # name, so repeated invocations (even across model switches) are cheap.
    model = _get_model(model_name)

    # --- Step (b): encode the job description ------------------------------
    print("Encoding JD...")
    try:
        # encode() on a single string returns a 1-D numpy array of shape
        # (vector_dimensions,) — e.g. (384,) for all-MiniLM-L6-v2.
        jd_vector = model.encode(
            job_description,
            show_progress_bar=True,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to encode the job description with '{model_name}': {e}"
        ) from e

    # --- Step (c): encode candidate profiles in memory-friendly CHUNKS ------
    # Encoding all 100K profiles in one encode() call can exhaust RAM because
    # sentence-transformers tokenises the full list up-front.  Instead we
    # split into chunks of CHUNK_SIZE, encode each chunk, and stitch the
    # resulting numpy arrays together at the end.  This caps peak memory at
    # roughly (CHUNK_SIZE × avg_tokens × model_dims) instead of (N × ...).
    CHUNK_SIZE = 5_000  # 5 000 profiles per chunk — safe for 8 GB machines
    print(f"Encoding {len(profile_texts)} candidates (in chunks of {CHUNK_SIZE})...")
    try:
        chunk_vectors = []
        for chunk_start in range(0, len(profile_texts), CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, len(profile_texts))
            chunk_num = chunk_start // CHUNK_SIZE + 1
            total_chunks = (len(profile_texts) + CHUNK_SIZE - 1) // CHUNK_SIZE
            print(f"  Chunk {chunk_num}/{total_chunks} "
                  f"(candidates {chunk_start+1}–{chunk_end})...")
            vectors = model.encode(
                profile_texts[chunk_start:chunk_end],
                show_progress_bar=True,
                batch_size=64,
            )
            chunk_vectors.append(vectors)
        # Stack all chunks into one (N, dims) array
        candidate_vectors = np.vstack(chunk_vectors)
        del chunk_vectors  # free the duplicate memory immediately
    except Exception as e:
        raise RuntimeError(
            f"Failed to encode candidate profiles with '{model_name}': {e}"
        ) from e

    # --- Step (d): compute cosine similarities -----------------------------
    print("Computing similarities...")
    try:
        # Imported here (not at top) to keep the module importable even if
        # scikit-learn somehow isn't present yet — the failure message is clearer.
        from sklearn.metrics.pairwise import cosine_similarity

        # cosine_similarity expects 2-D arrays of shape (n_samples_a, n_features)
        # and (n_samples_b, n_features). Our JD vector is 1-D (e.g. (384,)), so
        # we reshape it to (1, 384) — a "batch" of one job description.
        # candidate_vectors is already (N, dims) so no reshape needed.
        #
        # The result `sims` has shape (1, N): one row (the JD) vs N columns
        # (the candidates). sims[0, i] is the similarity between the JD and
        # candidate i.
        sims = cosine_similarity(jd_vector.reshape(1, -1), candidate_vectors)
    except Exception as e:
        raise RuntimeError(
            f"Failed to compute cosine similarities: {e}"
        ) from e

    # --- Step (e): build the {candidate_id: score} result dict -------------
    # sims[0] is the row vector of length N — the JD's similarity to each
    # candidate, in the same order as candidate_ids (because we built
    # profile_texts in that order back at the top).
    #
    # cosine_similarity can in rare cases return values fractionally outside
    # [0, 1] due to floating-point error, so we clip to be safe. (Values can
    # in principle be negative for unrelated text, but the spec wants 0..1.)
    scores = {}
    for i, cid in enumerate(candidate_ids):
        raw = float(sims[0, i])
        clipped = max(0.0, min(1.0, raw))
        # If the profile was empty/invalid, force 0.0 regardless of math.
        if not (isinstance(candidate_profiles[cid], str)
                and candidate_profiles[cid].strip()):
            clipped = 0.0
        scores[cid] = round(clipped, 4)

    return scores


# ===========================================================================
# HELPER FUNCTION — get_model_info()
# ===========================================================================
def get_model_info(model_name: str = None) -> dict:
    """
    Return basic info about a model this module can use.

    Parameters
    ----------
    model_name : str or None
        Which model to describe. If None, describes MODEL_NAME (the default).

    Returns
    -------
    dict with keys:
        model_name         -> the resolved model name
        vector_dimensions  -> its embedding size (looked up from SUPPORTED_MODELS,
                              so this is CORRECT per model, never a stale constant)
        speed              -> "fast" / "slow" label
        accuracy           -> "good" / "high" label

    Useful for the metadata YAML (output_generator.generate_metadata) and for
    debugging ("wait, which model are we running?").

    NOTE: dimensions are read from SUPPORTED_MODELS, NOT queried from a live
    model — that would force a model load just to read a constant. Keep the
    registry in sync with reality when you add a model.
    """
    name = model_name if model_name is not None else MODEL_NAME
    info = SUPPORTED_MODELS.get(name, {})
    return {
        "model_name": name,
        "vector_dimensions": info.get("vector_dimensions", "unknown"),
        "speed": info.get("speed", "unknown"),
        "accuracy": info.get("accuracy", "unknown"),
    }


# ===========================================================================
# REAL-DATA LOADERS
# ===========================================================================
# These load the teammate-provided REAL sample data so we can test the scorer
# on actual profile shapes (not just the synthetic TEST_CANDIDATES). They are
# kept separate from the core scorer so other modules can import them too.
#
# IMPORTANT context about the real sample: the 5 real candidates are a Backend
# Engineer, Operations Manager, Customer Support, Marketing Manager, and
# Accountant. NONE of them is a Senior AI Engineer (what the JD asks for). So
# a "correct" result here is NOT "everyone scores high" — it's "the Backend /
# data engineer scores highest, the rest drop off." Keep this in mind when
# reading the test output: low absolute scores on real data are EXPECTED.

def load_real_sample_candidates(filepath=None):
    """
    Load the teammate-provided real sample candidates and return them in the
    {candidate_id: profile_text} shape that compute_semantic_scores() expects.

    The file is a JSON list of candidate dicts. We use:
      - "candidate_id"  -> the ID key (spec-compliant 7-digit format, e.g.
                           CAND_0000001). We trust the teammate's field name
                           rather than guessing — it matches candidate_schema.json
                           and validate_submission.py.
      - "profile_text"  -> the ID-and-text combined profile the teammate already
                           built. We use it directly rather than re-combining
                           fields ourselves, because (a) the teammate already
                           made the right call about which sections to include,
                           and (b) it keeps our profile shape identical to what
                           the real pipeline will feed in.

    Why not combine headline+summary+skills+experience ourselves? Because the
    teammate's profile_text is the CANONICAL input for the real pipeline — using
    anything else here would test a different code path than production runs.
    We DO log which field we picked and a preview, so it's transparent.

    Parameters
    ----------
    filepath : str or Path or None
        Where to load from. If None, uses REAL_SAMPLE_FILE (data/sample_5_candidates.json).

    Returns
    -------
    dict
        {candidate_id: profile_text} for each candidate in the file.

    Raises
    ------
    FileNotFoundError
        If the file doesn't exist (with a hint to copy it from the teammate's zip).
    ValueError
        If the file parses but doesn't have the fields we need.
    """
    path = Path(filepath) if filepath is not None else REAL_SAMPLE_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Real sample file not found: {path}\n"
            "Get it from your teammate's candidate-discovery-engine.zip and "
            "copy data/sample_5_candidates.json into the data/ folder."
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse JSON in {path}: {e}") from e

    # The file should be a list of candidate dicts. Validate the shape before
    # we touch it so the error message is helpful.
    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON list in {path}, got {type(data).__name__}."
        )

    profiles = {}
    missing_id = []
    missing_text = []
    for entry in data:
        # Identify the candidate ID. We require "candidate_id" (the schema field).
        # If a future teammate file uses a different key, this surfaces it clearly.
        cid = entry.get("candidate_id")
        if not cid:
            missing_id.append(entry)
            continue
        # Prefer the pre-combined profile_text; fall back to summary if absent
        # (some slimmed-down samples might omit the combined field).
        text = entry.get("profile_text") or entry.get("summary") or ""
        if not text.strip():
            missing_text.append(cid)
            text = ""
        profiles[str(cid)] = text

    if missing_id:
        print(f"⚠️  Skipped {len(missing_id)} candidate(s) with no candidate_id field.")
    if missing_text:
        print(f"⚠️  These candidates had no usable profile text (will score 0.0): "
              f"{missing_text}")

    print(f"Loaded {len(profiles)} real candidates from {path.name} "
          f"(using fields: candidate_id + "
          f"{'profile_text' if data and data[0].get('profile_text') else 'summary'}).")
    return profiles


def load_job_description_from_docx(filepath=None) -> str:
    """
    Extract plain text from the real job_description.docx.

    The .docx is a Word document with 68 paragraphs of real JD text (~9.5k
    chars). We concatenate the non-empty paragraphs with newlines and return
    the result for use as the job_description argument.

    python-docx is already in requirements.txt (as `python-docx`), imported
    as `docx`. We import it lazily so importing this module never forces a
    docx dependency unless you actually call this function.

    Parameters
    ----------
    filepath : str or Path or None
        Defaults to data/job_description.docx.

    Returns
    -------
    str
        The full JD text.

    Raises
    ------
    FileNotFoundError
        If the docx isn't there. The caller (test block) is expected to fall
        back to TEST_JOB_DESCRIPTION when this happens.
    """
    path = Path(filepath) if filepath is not None else JOB_DESCRIPTION_DOCX
    if not path.exists():
        raise FileNotFoundError(
            f"Job description .docx not found: {path}. "
            "Falling back to TEST_JOB_DESCRIPTION."
        )

    # Lazy import so the module is importable even without python-docx.
    try:
        import docx
    except ImportError as e:
        raise ImportError(
            "python-docx is required to read the JD .docx. "
            "Install it: pip install python-docx\n"
            f"(original error: {e})"
        ) from e

    document = docx.Document(str(path))

    # Pull every non-empty paragraph. We skip blanks so the model doesn't
    # waste embedding capacity on whitespace. (Tables aren't expected in this
    # JD, but if there were any, this loop would miss them — acceptable here.)
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    full_text = "\n".join(paragraphs)

    if not full_text.strip():
        raise ValueError(f"No text found in {path} (empty document?).")

    print(f"Loaded real job description from {path.name} "
          f"({len(paragraphs)} paragraphs, {len(full_text)} chars).")
    return full_text


# ===========================================================================
# TEST HELPERS (used only by the __main__ comparison block below)
# ===========================================================================
def _rank_map(scores: dict) -> dict:
    """
    Turn a {candidate_id: score} dict into {candidate_id: rank}, where rank 1
    is the highest score. Ties are broken by candidate_id ascending (matching
    the submission spec's tie-break rule) so ranks are deterministic.
    """
    # Sort by (-score, id): highest score first; on ties, smaller id first.
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return {cid: rank for rank, (cid, _) in enumerate(ordered, start=1)}


def _compare_and_recommend(label_a: str, res_a: dict,
                           label_b: str, res_b: dict,
                           time_a: float, time_b: float) -> str:
    """
    Print a side-by-side comparison of two model runs and recommend one.

    Each `res_*` is the dict returned by compute_semantic_scores:
        {"model_name": ..., "scores": {cid: score}, "ok": bool, "error": str}
    `time_*` is the wall-clock seconds that run took (end-to-end, including
    model load on first use).

    Returns the name of the recommended model (or label_a if b failed).

    The recommendation is a transparent heuristic, not magic:
      - The hackathon is a ONE-SHOT run over a fixed candidate set, so accuracy
        matters more than per-call speed (we only pay the cost once).
      - BUT if both models agree on the ranking, the faster one is good enough
        and we save time/risk.
      - "Meaningful disagreement" = a different candidate at rank 1, OR many
        rank swaps. In that case we trust the more accurate model.
    """
    # ---------------------------------------------------------------------
    # 1. Handle the case where one model failed to load/run.
    # ---------------------------------------------------------------------
    if not res_b["ok"]:
        print(f"\n⚠️  {label_b} failed ({res_b['error']}); cannot compare.")
        print(f"👉 Recommend {label_a} (it's the only one that ran).")
        return res_a["model_name"]
    if not res_a["ok"]:
        print(f"\n⚠️  {label_a} failed ({res_a['error']}); cannot compare.")
        print(f"👉 Recommend {label_b} (it's the only one that ran).")
        return res_b["model_name"]

    scores_a, scores_b = res_a["scores"], res_b["scores"]
    ranks_a, ranks_b = _rank_map(scores_a), _rank_map(scores_b)

    # All candidate ids (both dicts share the same keys; use scores_a's order).
    cids = list(scores_a.keys())

    # ---------------------------------------------------------------------
    # 2. Side-by-side scores + ranks, flagging any rank changes.
    # ---------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 78)
    header = (f"{'CANDIDATE':<12}{label_a:>20}{label_b:>20}   {'CHANGE'}")
    # Show score and rank under each model label for readability.
    print(f"{'CANDIDATE':<12}{'score / rank':>20}{'score / rank':>20}   {'(rank)'}")
    print("-" * 78)
    changed_ids = []
    for cid in cids:
        sa, sb = scores_a[cid], scores_b[cid]
        ra, rb = ranks_a[cid], ranks_b[cid]
        changed = ra != rb
        if changed:
            changed_ids.append(cid)
        delta = f"{ra} -> {rb}" if changed else "same"
        marker = " ◀ changed" if changed else ""
        print(f"{cid:<12}{sa:>8.4f} (r{ra}){sb:>8.4f} (r{rb})   {delta}{marker}")
    print("-" * 78)

    # ---------------------------------------------------------------------
    # 3. Which candidates changed rank between the two models?
    # ---------------------------------------------------------------------
    if changed_ids:
        print(f"\nCandidates whose rank changed: {len(changed_ids)} / {len(cids)}")
        for cid in changed_ids:
            print(f"  • {cid}: {label_a} rank {ranks_a[cid]}  ->  "
                  f"{label_b} rank {ranks_b[cid]}")
    else:
        print("\nNo rank changes — both models ordered the candidates identically.")

    # Did the #1 pick change? That's the most consequential difference.
    # We read the rank-1 candidate from each rank map (which already applies
    # the spec's tie-break-by-id rule), so ties are handled deterministically.
    top_a = next(c for c, r in ranks_a.items() if r == 1)
    top_b = next(c for c, r in ranks_b.items() if r == 1)
    top1_changed = top_a != top_b
    print(f"\nTop candidate:  {label_a} -> {top_a}   |   {label_b} -> {top_b}"
          + ("   ⚠️ DIFFERENT" if top1_changed else "   (same)"))

    # ---------------------------------------------------------------------
    # 4. Timing.
    # ---------------------------------------------------------------------
    print("\nTiming (end-to-end, including model load on first use):")
    print(f"  {label_a}: {time_a:.2f}s")
    print(f"  {label_b}: {time_b:.2f}s")
    ratio = time_b / time_a if time_a > 0 else float("inf")
    faster_label = label_a if time_a <= time_b else label_b
    print(f"  {faster_label} was faster; {label_b} took {ratio:.1f}x as long "
          f"as {label_a}." if time_b >= time_a
          else f"  {faster_label} was faster; {label_a} took {1/ratio:.1f}x "
               f"as long as {label_b}.")
    print("  (In production the model loads ONCE, so the per-call encode time "
          "gap is smaller than these totals suggest.)")

    # ---------------------------------------------------------------------
    # 5. Recommendation.
    # ---------------------------------------------------------------------
    # Heuristic, in priority order (see docstring for rationale):
    bge_label = "BAAI/bge-base-en-v1.5"
    minilm_label = "sentence-transformers/all-MiniLM-L6-v2"
    # Figure out which physical label is the "accurate/slow" one.
    accurate_label = label_b if res_b["model_name"] == bge_label else label_a
    fast_label = label_a if res_a["model_name"] == minilm_label else label_b

    print("\n" + "=" * 78)
    print("RECOMMENDATION FOR FINAL SUBMISSION")
    print("=" * 78)

    if top1_changed:
        # The two models disagree on the single most important slot. For a
        # one-shot submission we trust the more accurate model.
        print(f"The two models DISAGREE on the #1 candidate. For a one-shot")
        print(f"final submission, accuracy matters more than speed, so we trust")
        print(f"the higher-accuracy model.")
        print(f"\n👉 Use: {accurate_label}")
        return res_b["model_name"] if accurate_label == label_b else res_a["model_name"]

    if not changed_ids:
        # Identical ranking. No reason to pay for the bigger model.
        print(f"Both models produced the IDENTICAL ranking. The larger model")
        print(f"adds no value here, so go with the faster, lighter one.")
        print(f"\n👉 Use: {fast_label}")
        return res_a["model_name"] if fast_label == label_a else res_b["model_name"]

    # Some ranks shuffled, but the top pick held. Lean accurate for the final
    # submission (it's run once), unless the gap is tiny and bge was vastly
    # slower — but we still default to accuracy-first here.
    print(f"Rankings are CLOSE but not identical ({len(changed_ids)} candidate(s) ")
    print(f"moved). The top candidate agreed. For a one-shot final submission we")
    print(f"prefer the higher-accuracy model so the mid-tier ordering is as sharp")
    print(f"as possible — the cost is paid only once.")
    print(f"\n👉 Use: {accurate_label}")
    print(f"   (Switch to {fast_label} if download size or runtime is a blocker.)")
    return res_b["model_name"] if accurate_label == label_b else res_a["model_name"]


# ===========================================================================
# MAIN — TEST MODE (runs BOTH models and compares)
# ===========================================================================
# Runs only when you execute this file directly (NOT when it's imported by
# score_combiner). It exercises the whole module on the built-in test data
# using BOTH supported models, then prints a side-by-side comparison and a
# recommendation for which to use in the final submission.
if __name__ == "__main__":
    print("=" * 60)
    print("SEMANTIC SCORER - TEST MODE (dual-model comparison)")
    print("=" * 60)

    MINILM = "sentence-transformers/all-MiniLM-L6-v2"
    BGE = "BAAI/bge-base-en-v1.5"

    print(f"\nModels under test:")
    for m in (MINILM, BGE):
        info = get_model_info(m)
        print(f"  • {m}  [{info['speed']}/{info['accuracy']} accuracy, "
              f"{info['vector_dimensions']}-dim]")
    print(f"Test set: {len(TEST_CANDIDATES)} candidates\n")

    # Helper to run one model end-to-end and capture timing + a clean result
    # envelope (so a failure in one model doesn't abort the whole comparison).
    def _run_one(model_name):
        t0 = time.perf_counter()
        try:
            scores = compute_semantic_scores(
                TEST_JOB_DESCRIPTION, TEST_CANDIDATES,
                model_name=model_name,
            )
            elapsed = time.perf_counter() - t0
            return {"model_name": model_name, "scores": scores,
                    "ok": True, "error": None, "time": elapsed}
        except Exception as e:
            elapsed = time.perf_counter() - t0
            return {"model_name": model_name, "scores": {},
                    "ok": False, "error": str(e), "time": elapsed}

    # --- Run model A (fast) ------------------------------------------------
    print("\n" + "#" * 60)
    print(f"# MODEL A: {MINILM}")
    print("#" * 60)
    res_minilm = _run_one(MINILM)

    # --- Run model B (accurate) -------------------------------------------
    # Heads-up: this downloads ~420MB the first time.
    print("\n" + "#" * 60)
    print(f"# MODEL B: {BGE}  (first run downloads ~420MB)")
    print("#" * 60)
    res_bge = _run_one(BGE)

    # --- If at least one ran, show its ranking -----------------------------
    for label, res in (("MINILM", res_minilm), ("BGE", res_bge)):
        if not res["ok"]:
            print(f"\n❌ {label} failed: {res['error']}")
            continue
        ranked = sorted(res["scores"].items(), key=lambda kv: kv[1], reverse=True)
        print(f"\n--- {label} ranking ({res['time']:.2f}s) ---")
        for rank, (cid, score) in enumerate(ranked, start=1):
            preview = " ".join(TEST_CANDIDATES[cid].split())[:60]
            print(f"  {rank}. {cid}  {score:.4f}  {preview}")

    # --- Compare, time, and recommend -------------------------------------
    if res_minilm["ok"] or res_bge["ok"]:
        recommended = _compare_and_recommend(
            label_a="MINILM", res_a=res_minilm,
            label_b="BGE", res_b=res_bge,
            time_a=res_minilm["time"], time_b=res_bge["time"],
        )
        print(f"\n{'=' * 60}")
        print(f"RECOMMENDED FOR FINAL SUBMISSION: {recommended}")
        print(f"{'=' * 60}")
    else:
        print("\n❌ Both models failed — check the errors above.")
        sys.exit(1)

    # --- Sanity check against the original expectations --------------------
    # Independent of which model "won": a correctly-working scorer must put an
    # ML/AI candidate at the top and a non-technical candidate at the bottom.
    print("\n--- Sanity check (using the recommended model) ---")
    check_res = res_bge if recommended == BGE else res_minilm
    if check_res["ok"]:
        ranked = sorted(check_res["scores"].items(),
                        key=lambda kv: kv[1], reverse=True)
        top_id, bottom_id = ranked[0][0], ranked[-1][0]
        all_in_range = all(0.0 <= s <= 1.0 for s in check_res["scores"].values())
        if top_id in ("CAND_001", "CAND_005"):
            print(f"✅ Top candidate is {top_id} (ML/AI) — looks right.")
        else:
            print(f"⚠️  Top candidate is {top_id}, expected CAND_001 or CAND_005.")
        if bottom_id in ("CAND_002", "CAND_004"):
            print(f"✅ Bottom candidate is {bottom_id} (non-technical) — looks right.")
        else:
            print(f"⚠️  Bottom candidate is {bottom_id}, expected CAND_002 or CAND_004.")
        print(f"{'✅' if all_in_range else '❌'} All scores in [0, 1]: {all_in_range}.")

    # =====================================================================
    # PART 2: REAL SAMPLE DATA TEST
    # =====================================================================
    # Now run the recommended model against the teammate-provided REAL sample
    # candidates and the REAL job description (extracted from .docx). This
    # verifies the scorer works on the actual profile shapes the pipeline
    # will see, not just our synthetic test data.
    #
    # IMPORTANT — how to read these results:
    # The 5 real candidates are a Backend Engineer, Operations Manager,
    # Customer Support, Marketing Manager, and Accountant. The JD asks for a
    # Senior AI Engineer. So NOBODY is a perfect match; absolute scores will
    # be lower than on the fake data, and that's CORRECT. The signal we want
    # is *relative* ordering: the Backend Engineer (closest to data/ML work)
    # should rank highest.
    print("\n" + "=" * 60)
    print("PART 2: REAL SAMPLE DATA TEST")
    print("=" * 60)

    # --- Step 4: load the real job description ------------------------------
    # Try the .docx first; fall back to TEST_JOB_DESCRIPTION if it's missing
    # (e.g. someone cloned only src/). We treat a missing JD as a soft warning,
    # not a hard failure, so the test still runs.
    try:
        real_jd = load_job_description_from_docx()
        jd_source = "job_description.docx (REAL)"
    except Exception as e:
        print(f"\n⚠️  Could not load real JD ({e}); using TEST_JOB_DESCRIPTION.")
        real_jd = TEST_JOB_DESCRIPTION
        jd_source = "TEST_JOB_DESCRIPTION (synthetic fallback)"

    # --- Steps 1-3: load the real sample candidates -------------------------
    # load_real_sample_candidates() figures out the ID field (candidate_id)
    # and text field (profile_text) and returns {candidate_id: profile_text}.
    try:
        real_profiles = load_real_sample_candidates()
    except Exception as e:
        print(f"\n❌ Could not load real sample candidates: {e}")
        print("Skipping the real-data test. The fake-data results above still "
              "stand.")
        sys.exit(0 if (res_minilm["ok"] or res_bge["ok"]) else 1)

    # Quick preview so the reader can see WHAT we're scoring.
    print(f"\nReal JD source: {jd_source}")
    print(f"Real candidates ({len(real_profiles)}):")
    for cid, text in real_profiles.items():
        # Show first line (usually the Headline) as a human-readable label.
        first_line = text.split("\n", 1)[0][:60]
        print(f"  • {cid}  ({len(text)} chars)  {first_line}")

    # --- Step 5: run semantic scoring on the real data ----------------------
    # We use the RECOMMENDED model from Part 1, so the real-data result mirrors
    # what the final pipeline would produce.
    print(f"\n--- Scoring real sample data with the recommended model "
          f"({recommended}) ---")
    try:
        t0 = time.perf_counter()
        real_scores = compute_semantic_scores(
            real_jd, real_profiles, model_name=recommended,
        )
        real_elapsed = time.perf_counter() - t0
    except Exception as e:
        print(f"\n❌ Real-data scoring failed: {e}")
        sys.exit(1)

    # --- Step 6: print real results + compare with fake-data results --------
    real_ranked = sorted(real_scores.items(), key=lambda kv: kv[1], reverse=True)

    print(f"\n--- REAL sample ranking ({real_elapsed:.2f}s) ---")
    for rank, (cid, score) in enumerate(real_ranked, start=1):
        preview = " ".join(real_profiles[cid].split())[:60]
        print(f"  {rank}. {cid}  {score:.4f}  {preview}")

    # Side-by-side: fake-data best scores vs real-data scores. The point is to
    # show that the model is calibrated — real candidates who don't match the
    # role score lower than the (well-matched) fake ML candidates.
    print("\n--- Score comparison: fake (matching) data vs real sample ---")
    fake_best = res_bge["scores"] if recommended == BGE else res_minilm["scores"]
    fake_best_sorted = sorted(fake_best.values(), reverse=True)
    real_sorted = sorted(real_scores.values(), reverse=True)
    print(f"  {'DATASET':<22}{'TOP':>8}{'MEDIAN':>8}{'BOTTOM':>8}{'ALL 0-1?':>10}")
    print(f"  {'-' * 56}")
    fake_ok = all(0.0 <= s <= 1.0 for s in fake_best.values())
    real_ok = all(0.0 <= s <= 1.0 for s in real_scores.values())
    print(f"  {'fake TEST_CANDIDATES':<22}{fake_best_sorted[0]:>8.4f}"
          f"{fake_best_sorted[len(fake_best_sorted)//2]:>8.4f}"
          f"{fake_best_sorted[-1]:>8.4f}{'  ✅' if fake_ok else '  ❌':>10}")
    print(f"  {'real sample (5)':<22}{real_sorted[0]:>8.4f}"
          f"{real_sorted[len(real_sorted)//2]:>8.4f}"
          f"{real_sorted[-1]:>8.4f}{'  ✅' if real_ok else '  ❌':>10}")

    # --- Real-data sanity check --------------------------------------------
    # As discussed at the top of the real-data loaders: the Backend Engineer
    # (CAND_0000001, a data/ML-adjacent role) should be the best match for an
    # AI Engineer JD among this particular sample of non-AI candidates.
    print("\n--- Real-data sanity check ---")
    real_top_id = real_ranked[0][0]
    real_bottom_id = real_ranked[-1][0]
    if real_top_id == "CAND_0000001":
        print(f"✅ Top real candidate is {real_top_id} (Backend / data eng) — "
              f"the closest role to AI Engineering in this sample. Looks right.")
    else:
        print(f"⚠️  Top real candidate is {real_top_id}; expected CAND_0000001 "
              f"(Backend Engineer). Worth a manual look.")
    print(f"ℹ️  Bottom real candidate is {real_bottom_id} (lowest semantic "
          f"overlap with the AI Engineer JD).")
    print(f"{'✅' if real_ok else '❌'} All real scores in [0, 1]: {real_ok}.")
    print("\nNote: absolute scores on the real sample are expectedly LOWER than")
    print("on the fake data — none of these 5 candidates is actually an AI")
    print("engineer. The meaningful signal is the RELATIVE ranking, and that")
    print("the Backend Engineer sits on top.")
    print("\n✅ Real-data test complete.")
