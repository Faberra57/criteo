from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.annotation_tool import (  # noqa: E402
    AnnotationConfig,
    clear_row_annotation,
    ensure_annotation_columns,
    ensure_backup,
    filter_annotation_indices,
    format_row_for_terminal,
    load_annotation_frame,
    mark_row,
    parse_candidates,
    save_annotation_frame,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive terminal annotation tool for L2 labels."
    )
    parser.add_argument(
        "--input-path",
        default="data/real_l2_annotation_sample_llm_targeted.csv",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="If omitted, annotations are saved back into the input file.",
    )
    parser.add_argument(
        "--level1",
        default=None,
        help="Optional filter on one level_1_name value.",
    )
    parser.add_argument(
        "--statuses",
        nargs="+",
        default=["todo", "review"],
        help="Statuses to include in the annotation session.",
    )
    parser.add_argument(
        "--start-at",
        type=int,
        default=1,
        help="1-based position inside the filtered subset.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path) if args.output_path else input_path
    config = AnnotationConfig()

    df = load_annotation_frame(input_path)
    df = ensure_annotation_columns(df, config)
    backup_path = ensure_backup(input_path, output_path)

    filtered_indices = filter_annotation_indices(
        df,
        config=config,
        level1_value=args.level1,
        statuses=args.statuses,
    )
    if not filtered_indices:
        print("No rows match the requested filters.")
        return

    pointer = max(0, min(len(filtered_indices) - 1, args.start_at - 1))

    print("Interactive L2 annotation session")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    if backup_path:
        print(f"Backup created: {backup_path}")
    print(f"Rows in session: {len(filtered_indices)}")
    if args.level1:
        print(f"Filtered level 1: {args.level1}")
    print(f"Included statuses: {', '.join(args.statuses)}")
    print("")

    while 0 <= pointer < len(filtered_indices):
        row_index = filtered_indices[pointer]
        row = df.loc[row_index]
        print(
            format_row_for_terminal(
                row,
                position=pointer + 1,
                total=len(filtered_indices),
                config=config,
            )
        )
        raw_choice = input("\nChoice: ").strip()

        if not raw_choice:
            continue

        lowered = raw_choice.lower()
        if lowered == "q":
            save_annotation_frame(df, output_path)
            print(f"Progress saved to {output_path}")
            return

        if lowered == "p":
            pointer = max(0, pointer - 1)
            continue

        if lowered == "n":
            note_value = input("New note: ").strip()
            mark_row(df, row_index, config=config, notes=note_value)
            save_annotation_frame(df, output_path)
            continue

        if lowered == "c":
            clear_row_annotation(df, row_index, config=config)
            save_annotation_frame(df, output_path)
            continue

        if lowered in {"s", "r"}:
            note_value = input("Optional note: ").strip()
            new_status = "skipped" if lowered == "s" else "review"
            mark_row(
                df,
                row_index,
                config=config,
                notes=note_value,
                status=new_status,
            )
            save_annotation_frame(df, output_path)
            pointer += 1
            continue

        candidates = parse_candidates(row.get(config.candidates_col, ""))
        if raw_choice.isdigit():
            selected_idx = int(raw_choice) - 1
            if 0 <= selected_idx < len(candidates):
                selected_label = candidates[selected_idx]
                existing_note_value = row.get(config.notes_col, "")
                if existing_note_value != existing_note_value:
                    existing_note = ""
                else:
                    existing_note = str(existing_note_value or "").strip()
                note_value = input(
                    f"Note for '{selected_label}' [{existing_note}]: "
                ).strip()
                final_note = note_value if note_value else existing_note
                mark_row(
                    df,
                    row_index,
                    config=config,
                    label=selected_label,
                    notes=final_note,
                    status="annotated",
                )
                save_annotation_frame(df, output_path)
                pointer += 1
                continue

        print("Invalid input. Use a candidate number or one of the commands shown above.\n")

    save_annotation_frame(df, output_path)
    print(f"Session complete. Progress saved to {output_path}")


if __name__ == "__main__":
    main()
