"""Cleaning report generation and management."""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional


def save_cleaning_report(report: Dict[str, Any], output_path: str):
    """Save cleaning report as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Cleaning report saved: {output_path}")


def save_cleaning_summary(report: Dict[str, Any], output_path: str):
    """Save human-readable cleaning summary."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=" * 60)
    lines.append("CORPUS CLEANING REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Input file:  {report.get('input_file', 'N/A')}")
    lines.append(f"Output file: {report.get('output_file', 'N/A')}")
    lines.append(f"Text key:    {report.get('text_key', 'N/A')}")
    lines.append("")
    lines.append(f"Total input:     {report['total_input']:,} documents")
    lines.append(f"Total output:    {report['total_output']:,} documents")
    lines.append(f"Total filtered:  {report['total_filtered']:,} ({report['filtered_percentage']:.1f}%)")
    lines.append(f"Processing time: {report['total_time_seconds']:.1f}s")
    lines.append(f"Throughput:      {report['throughput_docs_per_sec']:.0f} docs/s")
    lines.append("")
    lines.append("Removal reasons:")
    lines.append("-" * 40)
    for reason, count in report.get("removal_reasons", {}).items():
        pct = count / max(report["total_input"], 1) * 100
        lines.append(f"  {reason:40s} {count:>8,} ({pct:.1f}%)")

    # Quality summary
    quality = report.get("quality_summary", {})
    if quality:
        lines.append("")
        lines.append("Quality metrics summary:")
        lines.append("-" * 40)
        for key, stats in quality.items():
            lines.append(f"  {key:30s} mean={stats['mean']:.3f} min={stats['min']:.3f} max={stats['max']:.3f}")

    lines.append("")
    lines.append("=" * 60)

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Cleaning summary saved: {output_path}")


def merge_reports(reports: list) -> Dict[str, Any]:
    """Merge multiple cleaning reports into one."""
    if not reports:
        return {}

    merged = {
        "total_input": sum(r.get("total_input", 0) for r in reports),
        "total_output": sum(r.get("total_output", 0) for r in reports),
        "total_filtered": sum(r.get("total_filtered", 0) for r in reports),
        "total_time_seconds": sum(r.get("total_time_seconds", 0) for r in reports),
        "removal_reasons": {},
        "filter_times_seconds": {},
        "quality_summary": {},
    }

    merged["filtered_percentage"] = round(
        (1 - merged["total_output"] / max(merged["total_input"], 1)) * 100, 2
    )
    merged["throughput_docs_per_sec"] = round(
        merged["total_input"] / max(merged["total_time_seconds"], 0.001), 1
    )

    # Merge removal reasons
    for report in reports:
        for reason, count in report.get("removal_reasons", {}).items():
            merged["removal_reasons"][reason] = merged["removal_reasons"].get(reason, 0) + count

    # Merge filter times
    for report in reports:
        for filt, t in report.get("filter_times_seconds", {}).items():
            merged["filter_times_seconds"][filt] = merged["filter_times_seconds"].get(filt, 0) + t

    # Sort removal reasons
    merged["removal_reasons"] = dict(
        sorted(merged["removal_reasons"].items(), key=lambda x: -x[1])
    )

    return merged
