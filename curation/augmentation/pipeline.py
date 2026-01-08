from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, Iterable, List, Tuple
import json
import threading
import time

from .data_models import AugmentedRecord, RejectedRecord, TemplateSpec, VariantRecord
from .generator import TeacherConverter, VariantGenerator, compute_deltas_for_variant
from .verifier import compile_verifier, run_verifier


class JsonlLiveWriter:
    """Thread-safe JSONL writer that flushes immediately."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._fh = open(path, "w", encoding="utf-8")

    def write(self, row: Dict[str, Any]) -> None:
        if not self._fh:
            return
        data = json.dumps(row, ensure_ascii=False)
        with self._lock:
            self._fh.write(data + "\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def augment_items(items: Iterable[Dict[str, Any]], args, teacher_oracle, judge_oracle,
                  live_accept_writer: JsonlLiveWriter | None = None,
                  live_reject_writer: JsonlLiveWriter | None = None):
    """Run the end-to-end augmentation loop for any seed dataset."""

    worker = partial(
        _process_one_item,
        args=args,
        teacher=teacher_oracle,
        judge=judge_oracle,
        accept_writer=live_accept_writer,
        reject_writer=live_reject_writer,
    )

    accepted_rows: List[Dict[str, Any]] = []
    rejected_rows: List[Dict[str, Any]] = []

    num_workers = getattr(args, "num_workers", 1) or 1
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        for result in ex.map(worker, items):
            rec = result.get("record") if isinstance(result, dict) else None
            if rec:
                accepted_rows.append(rec)
            rej = result.get("rejections") if isinstance(result, dict) else None
            if rej:
                rejected_rows.extend(rej)

    return accepted_rows, rejected_rows


def _process_one_item(it, args, teacher, judge, accept_writer=None, reject_writer=None):
    """Shared worker used by all dataset-specific CLI entrypoints."""

    target = args.variants_per_item
    max_gen_calls = args.max_gen_calls or (target * 50)

    feedback = None
    success = False
    variants: List[VariantRecord] = []
    ts: TemplateSpec | None = None
    rejections: List[Dict[str, Any]] = []

    def record_rejection(stage: str, reason_code: str, reason: str, candidate=None, meta=None):
        rec = RejectedRecord(
            id=it["id"],
            stage=stage,
            reason_code=reason_code,
            reason=reason,
            candidate=candidate,
            meta=meta or {}
        )
        rec_json = rec.to_json()
        rejections.append(rec_json)
        if reject_writer:
            reject_writer.write(rec_json)

    conv = TeacherConverter(teacher_oracle=teacher)

    per_retry_limit = getattr(args, "teacher_retry_time_limit_sec", 3600)
    if per_retry_limit is not None and per_retry_limit <= 0:
        per_retry_limit = None

    for attempt in range(args.teacher_retries):
        attempt_deadline = (time.monotonic() + per_retry_limit) if per_retry_limit else None
        timed_out = False
        print(f'Trying on {it["id"]}.. Attempt {attempt}...Start')
        try:
            ts = conv.convert_item(
                it,
                n_text_templates=args.n_text_templates,
                feedback=feedback
            )
            if attempt_deadline and time.monotonic() >= attempt_deadline:
                timed_out = True
                feedback = f"per-retry time limit ({per_retry_limit}s) exceeded after spec conversion."
                record_rejection(
                    stage="generation",
                    reason_code="per_retry_time_limit",
                    reason=feedback,
                    meta={"attempt": attempt, "time_limit_sec": per_retry_limit}
                )
                print(f'Trying on {it["id"]}.. Attempt {attempt}...Time limit reached post-conversion.')
                continue

            try:
                ver_fn = compile_verifier(ts.verifier.code)
            except Exception as e:
                print(f'Trying on {it["id"]}.. Attempt {attempt}...Verifier Error')
                feedback = f"verifier compilation failed:\n{type(e).__name__}: {e}"
                record_rejection(
                    stage="verifier_compile",
                    reason_code="verifier_compile_error",
                    reason=str(e),
                    meta={"attempt": attempt}
                )
                continue

            vg = VariantGenerator(lambda_text=args.lambda_text)
            try:
                vg._compile(ts)
            except Exception as e:
                print(f'Trying on {it["id"]}.. Attempt {attempt}...Generator Error')
                feedback = f"generator compilation failed:\n{type(e).__name__}: {e}"
                record_rejection(
                    stage="generator_compile",
                    reason_code="generator_compile_error",
                    reason=str(e),
                    meta={"attempt": attempt}
                )
                continue

            variants = []
            gen_calls = 0
            text_count = max(1, len(ts.text_templates))
            candidate_counter = 0

            while len(variants) < target and gen_calls < max_gen_calls:
                if attempt_deadline and time.monotonic() >= attempt_deadline:
                    timed_out = True
                    feedback = f"per-retry time limit ({per_retry_limit}s) exceeded while sampling."
                    record_rejection(
                        stage="generation",
                        reason_code="per_retry_time_limit",
                        reason=feedback,
                        meta={"attempt": attempt, "time_limit_sec": per_retry_limit, "generated": len(variants)}
                    )
                    print(f'Trying on {it["id"]}.. Attempt {attempt}...Time limit reached during generation.')
                    break
                seed = hash((it["id"], args.seed, attempt, gen_calls)) & 0x7FFFFFFF
                assign = vg.sample_assignment(ts, seed, max_tries=args.per_sample_max_tries)
                gen_calls += 1

                ok, y = run_verifier(ver_fn, assign)
                candidate_idx = candidate_counter
                candidate_counter += 1
                tid = candidate_idx % text_count
                x = vg.compose(ts.text_templates[tid], assign)
                candidate_payload = {
                    "seed": seed,
                    "assignment": assign,
                    "text_template_id": tid,
                    "x": x,
                    "y": y,
                }

                if not ok:
                    record_rejection(
                        stage="verifier_runtime",
                        reason_code="verifier_invalid_candidate",
                        reason="Sampled assignment failed verifier checks.",
                        candidate=candidate_payload,
                        meta={"attempt": attempt, "candidate_idx": candidate_idx}
                    )
                    continue

                dnum, dtext, dtotal, per = compute_deltas_for_variant(ts, vg, assign, tid)
                judge_example_id = f"{it['id']}::cand{candidate_idx}"
                try:
                    verdict = judge.review_candidate(
                        candidate_text=x,
                        answer=y,
                        example_id=judge_example_id,
                        source_question=it["question"],
                        source_answer=it["answer"],
                    )
                except Exception as e:
                    record_rejection(
                        stage="judge",
                        reason_code="judge_error",
                        reason=str(e),
                        candidate=candidate_payload,
                        meta={"candidate_id": judge_example_id}
                    )
                    continue

                if verdict.get("decision") == "accept":
                    variants.append(VariantRecord(
                        seed=seed,
                        assignment=assign,
                        text_template_id=tid,
                        x=x,
                        y=y,
                        delta_num=dnum,
                        delta_text=dtext,
                        delta_total=dtotal,
                        per_slot_delta=per,
                        judge_status="accept",
                        judge_category=verdict.get("category"),
                        judge_justification=verdict.get("justification"),
                    ))
                else:
                    reason = verdict.get("justification") or "Judge rejected candidate."
                    category = verdict.get("category") or "unspecified"
                    record_rejection(
                        stage="judge",
                        reason_code=f"judge_{category}",
                        reason=reason,
                        candidate=candidate_payload,
                        meta={"candidate_id": judge_example_id, "judge_raw": verdict.get("raw", verdict)}
                    )
                    continue

            if len(variants) >= target:
                success = True
                break

            if timed_out:
                continue

            feedback = (
                f"Insufficient judged-valid samples. Needed {target}, got {len(variants)}.\n"
                f"Please revise the generator/verifier so produced statements remain sound and pass verification."
            )
            record_rejection(
                stage="generation",
                reason_code="insufficient_variants",
                reason=f"Only {len(variants)} judged-valid variants (target {target}).",
                meta={"attempt": attempt, "generated": len(variants), "target": target}
            )
            print(f'Trying on {it["id"]}.. Attempt {attempt}...Low success rate.')

        except Exception as e:
            print(f'Trying on {it["id"]}.. Attempt {attempt}...Spec Conversion Failed...{e}')
            feedback = f"spec conversion failed:\n{type(e).__name__}: {e}"
            record_rejection(
                stage="spec",
                reason_code="spec_conversion_failed",
                reason=str(e),
                meta={"attempt": attempt}
            )
            continue

    if success and ts is not None:
        print(f'Success on {it["id"]}..')
        rec = AugmentedRecord(
            id=it["id"],
            source={"question": it["question"], "answer": it["answer"]},
            spec=ts,
            variants=variants
        )
        rec_json = rec.to_json()
        if accept_writer:
            accept_writer.write(rec_json)
        return {"record": rec_json, "rejections": rejections}

    print(f'Failure on {it["id"]}..')
    record_rejection(
        stage="pipeline",
        reason_code="max_teacher_retries",
        reason=f"max teacher retries reached; last feedback: {feedback}",
        meta={"teacher_retries": args.teacher_retries}
    )
    return {"record": None, "rejections": rejections, "error": feedback}
