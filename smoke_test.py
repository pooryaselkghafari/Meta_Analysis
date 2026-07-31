"""Smoke test: exercise the deterministic stages without Marker or the API.

Feeds a synthetic markdown 'paper' (with a regression table and prose) straight
into chunking + filtering, then checks the consistency validator.
"""
from meta_pipeline.models import ParsedPaper, ExtractionRecord, TransformationType
from meta_pipeline.config import ChunkConfig
from meta_pipeline import stage3_chunk

SAMPLE_MD = """
# 4. Results

We estimate the effect of monetary policy shocks on output. Our baseline
specification in Column 3 includes country and time fixed effects.

The table below reports the main results across specifications.

| Variable            | (1) OLS   | (2) FE    | (3) IV     |
|---------------------|-----------|-----------|------------|
| Policy shock        | -0.42***  | -0.38***  | -0.51***   |
|                     | (0.11)    | (0.09)    | (0.14)     |
| Inflation (control) | 0.05      | 0.03      | 0.04       |
| Country FE          | No        | Yes       | Yes        |
| Observations        | 1200      | 1200      | 1180       |
| R-squared           | 0.31      | 0.44      | 0.41       |

A one standard deviation policy shock reduces log output by roughly 0.5 percent
in our preferred IV specification. This effect is statistically significant.

# 5. Literature Review

Prior work by earlier authors discussed related mechanisms at length without
providing new estimates.
"""


def main():
    paper = ParsedPaper(paper_id="sample01", source_path="n/a", markdown=SAMPLE_MD)
    cfg = ChunkConfig()

    chunks = stage3_chunk.chunk_paper(paper, cfg)
    print(f"chunks produced: {len(chunks)}")
    for c in chunks:
        sig = stage3_chunk.heuristic_has_estimate(c, cfg)
        kind = "TABLE" if c.contains_table else "prose"
        print(f"  {c.chunk_id} [{kind}] estimate_signal={sig} len={len(c.text)}")

    # verify the table chunk carried its column headers + N + surrounding prose
    table_chunk = next(c for c in chunks if c.contains_table)
    assert "(3) IV" in table_chunk.text, "column headers lost"
    assert "Observations" in table_chunk.text, "bottom-of-table metadata lost"
    assert "Column 3" in table_chunk.text, "preceding context lost"
    assert "preferred IV specification" in table_chunk.text, "following context lost"
    print("OK: table chunk preserved headers, metadata, and surrounding context")

    # consistency validator
    rec = ExtractionRecord(paper_id="sample01", estimate_id="c001")
    rec.elasticity_is_raw = True
    rec.elasticity_transformation_type = TransformationType.LOG  # deliberate contradiction
    problems = rec.check_transformation_consistency()
    assert problems, "consistency check should have flagged the contradiction"
    print(f"OK: consistency validator caught contradiction -> {problems[0]}")

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
