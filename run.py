#!/usr/bin/env python3
"""CLI runner for the meta-analysis extraction pipeline.

Examples:
    # deterministic only (prompts still blank): parse + chunk + heuristic filter
    python run.py --input input_papers --output output

    # once prompts are written, enable LLM stages
    python run.py --input input_papers --output output --llm
"""
import argparse
import json

from meta_pipeline import Pipeline, PipelineConfig


def main():
    ap = argparse.ArgumentParser(description="Macro meta-analysis extraction pipeline")
    ap.add_argument("--input", default="input_papers", help="dir of PDF papers")
    ap.add_argument("--output", default="output", help="output dir")
    ap.add_argument("--targets", default="targets.json", help="user DV/IV targets")
    ap.add_argument("--ontology", default="ontology.json", help="variable ontology")
    ap.add_argument("--llm", action="store_true",
                    help="enable LLM stages (requires prompts + API key)")
    ap.add_argument("--force-ocr", action="store_true",
                    help="force OCR in Marker (scanned papers)")
    args = ap.parse_args()

    cfg = PipelineConfig(
        input_dir=args.input,
        output_dir=args.output,
        targets_path=args.targets,
        ontology_path=args.ontology,
        run_llm_stages=args.llm,
    )
    cfg.marker.force_ocr = args.force_ocr

    pipeline = Pipeline(cfg)
    summary = pipeline.run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
