"""Self-contained raw QED amplitude parser."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .interaction import QEDAmplitudeRecord


def _split_sections(raw: str) -> tuple[str, str, str, str]:
    raw = raw.strip()
    interaction_end = raw.find(" : Vertex")
    if interaction_end == -1:
        interaction_end = raw.find(":Vertex")
    if interaction_end == -1:
        raise ValueError("Could not find interaction/topology boundary.")

    sec1 = raw[:interaction_end].strip()
    rest = raw[interaction_end + 3 :].strip()

    v1_pos = rest.find("Vertex V_1")
    if v1_pos == -1:
        remaining_parts = rest.split(" : ", 2)
        sec2 = remaining_parts[0] if len(remaining_parts) > 0 else ""
        sec3 = remaining_parts[1] if len(remaining_parts) > 1 else ""
        sec4 = remaining_parts[2] if len(remaining_parts) > 2 else ""
        return sec1, sec2, sec3, sec4

    search_start = v1_pos + len("Vertex V_1")
    topology_end = rest.find(" : ", search_start)
    if topology_end == -1:
        return sec1, rest, "", ""

    sec2 = rest[:topology_end].strip()
    amplitude_and_squared = rest[topology_end + 3 :]
    last_colon = amplitude_and_squared.rfind(" : ")
    if last_colon == -1:
        return sec1, sec2, amplitude_and_squared.strip(), ""
    sec3 = amplitude_and_squared[:last_colon].strip()
    sec4 = amplitude_and_squared[last_colon + 3 :].strip()
    return sec1, sec2, sec3, sec4


def parse_record(
    line: str,
    *,
    source_file: str = "<memory>",
    source_line_index: int = 0,
) -> QEDAmplitudeRecord:
    sec1, sec2, sec3, sec4 = _split_sections(line)
    return QEDAmplitudeRecord(
        sample_id=f"{source_file}:{source_line_index}",
        source_file=source_file,
        source_line_index=source_line_index,
        raw_interaction=sec1,
        raw_topology=sec2,
        raw_amplitude=sec3,
        raw_squared=sec4,
    )


def parse_file(filepath: str | Path) -> list[QEDAmplitudeRecord]:
    filepath = Path(filepath)
    records: list[QEDAmplitudeRecord] = []
    with open(filepath, "r") as handle:
        for line_index, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(
                    parse_record(
                        line,
                        source_file=filepath.name,
                        source_line_index=line_index,
                    )
                )
            except Exception as exc:
                print(f"Warning: failed to parse QED line in {filepath.name}: {exc}")
    return records


@lru_cache(maxsize=None)
def _parse_all_qed_cached(data_dir_str: str) -> tuple[QEDAmplitudeRecord, ...]:
    data_dir = Path(data_dir_str)
    all_records: list[QEDAmplitudeRecord] = []
    for filepath in sorted(data_dir.glob("QED-2-to-2-diag-TreeLevel-*.txt")):
        records = parse_file(filepath)
        all_records.extend(records)
        print(f"  Parsed {filepath.name}: {len(records)} diagrams")
    print(f"Total QED diagrams: {len(all_records)}")
    return tuple(all_records)


def parse_all_qed(data_dir: str | Path) -> list[QEDAmplitudeRecord]:
    normalized = str(Path(data_dir).expanduser().resolve())
    return list(_parse_all_qed_cached(normalized))
