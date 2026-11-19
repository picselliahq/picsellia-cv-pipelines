#!/usr/bin/env python3
"""Read-only listing of the datasets and models of a Picsellia organization.

Used to pick the source data for the CI fixtures (e.g. the prod "demo-account"
organization). Nothing is created, modified, downloaded or deleted.

Usage:
    PXL_API_TOKEN_PROD=... uv run python scripts/list_org_resources.py \
        --organization demo-account --host https://app.picsellia.com \
        [--markdown docs/DEMO_ACCOUNT_RESOURCES.md]

The token is read from the environment variable named by --token-env
(default PXL_API_TOKEN_PROD) so it never lands in the shell history or in
process listings.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

COUNT_HINTS = ("count", "nb_", "size", "total")


def safe(fn, default="?"):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 - listing must keep going
        return f"{default} ({type(e).__name__})"


def dataset_version_row(dv) -> dict:
    stats = safe(dv.retrieve_stats, None)
    nb_ann = getattr(stats, "nb_annotations", "?") if stats and not isinstance(stats, str) else "?"
    labels = safe(lambda: sorted(label.name for label in dv.list_labels()), [])
    raw = safe(dv.sync, {})
    counts = (
        {k: v for k, v in raw.items() if any(h in k.lower() for h in COUNT_HINTS) and isinstance(v, int | float)}
        if isinstance(raw, dict)
        else {}
    )
    return {
        "name": dv.name,
        "version_id": str(dv.id),
        "type": str(getattr(dv, "type", "?")),
        "annotations": nb_ann,
        "labels": labels if isinstance(labels, list) else [],
        "counts": counts,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--organization", required=True)
    ap.add_argument("--host", default="https://app.picsellia.com")
    ap.add_argument("--token-env", default="PXL_API_TOKEN_PROD")
    ap.add_argument("--markdown", help="also write the listing as a markdown file")
    args = ap.parse_args()

    token = os.environ.get(args.token_env)
    if not token:
        print(f"environment variable {args.token_env} is required", file=sys.stderr)
        return 2

    from picsellia import Client

    client = Client(api_token=token, organization_name=args.organization, host=args.host)
    md = [f"# {args.organization} ({args.host})", "", "## Datasets", ""]

    print(f"\n=== Datasets of {args.organization} ===")
    for ds in client.list_datasets():
        print(f"\n{ds.name}")
        md += [f"### {ds.name}", "", "| Version | Type | Annotations | Labels | Counts | ID |", "|---|---|---|---|---|---|"]
        for dv in ds.list_versions():
            r = dataset_version_row(dv)
            labels = ", ".join(r["labels"][:8]) + (" …" if len(r["labels"]) > 8 else "")
            print(f"  - {r['name']:<28} {r['type']:<22} ann={r['annotations']}  labels=[{labels}]  {r['counts']}  {r['version_id']}")
            md.append(f"| {r['name']} | {r['type']} | {r['annotations']} | {labels} | {r['counts']} | `{r['version_id']}` |")
        md.append("")

    print(f"\n=== Models of {args.organization} ===")
    md += ["## Models", "", "| Model | Version | Type | Framework | Files | ID |", "|---|---|---|---|---|---|"]
    for model in client.list_models():
        for mv in model.list_versions():
            files = safe(lambda mv=mv: [f.name for f in mv.list_files()], [])
            files = files if isinstance(files, list) else []
            fw = getattr(mv, "framework", "?")
            tp = getattr(mv, "type", "?")
            print(f"  - {model.name} / {mv.name}  type={tp}  framework={fw}  files={files}  {mv.id}")
            md.append(f"| {model.name} | {mv.name} | {tp} | {fw} | {', '.join(files)} | `{mv.id}` |")

    if args.markdown:
        Path(args.markdown).write_text("\n".join(md) + "\n")
        print(f"\nlisting written to {args.markdown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
