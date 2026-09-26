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

# Flattened OCR-style table: each cell is its own short paragraph (no pipes).
# Must stay ONE chunk — headers + coefficients together — or detection fails.
FLATTENED_TABLE_MD = """
# Results

Estimates are reported below.

Table 2. Seemingly unrelated regression parameter estimate for share equations

Output Share Equations

Cereals

Pulses

Fruits &
Vegetables
0.041***
(5.58)

Animal
Products
0.009
(0.68)

Explanatory
Variables

Constant

-0.516

0.139

Time

0.005***
(3.65)

-0.031***
(-4.54)

System R2
Note: values in parentheses are t-statistics.

Discussion of the results continues here with a long prose paragraph that
should not be absorbed into the table body itself.
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

    flat_paper = ParsedPaper(paper_id="flat01", source_path="n/a", markdown=FLATTENED_TABLE_MD)
    flat_chunks = stage3_chunk.chunk_paper(flat_paper, cfg)
    flat_tables = [c for c in flat_chunks if c.contains_table]
    assert len(flat_tables) >= 1, f"expected flattened table chunk(s), got {len(flat_tables)}"
    # Caption + coefficients must share one chunk (the failure mode was
    # headers in chunk A and bare numbers in chunk B).
    ft = next(c for c in flat_tables if "Table 2. Seemingly" in c.text)
    assert "0.041***" in ft.text and "(5.58)" in ft.text and "System R2" in ft.text
    assert ft.text.index("Table 2. Seemingly") < ft.text.index("0.041***")
    orphan_coef_chunks = [
        c for c in flat_chunks
        if c is not ft and "0.041***" in c.text and "Table 2. Seemingly" not in c.text
    ]
    assert not orphan_coef_chunks, "flattened table coefficients were split from their caption"
    print("OK: flattened OCR-style table kept as one atomic chunk")

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
