from pathlib import Path
import json
import shutil

# formatting helpers


def _fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.2f}" if v > 50 else (f"{v:.4f}" if v < 10 else f"{v:.3f}")
    return str(v)


def _mid_trunc(s: str, width: int) -> str:
    if s is None:
        return "-"
    if width <= 3 or len(s) <= width:
        return s
    keep = width - 1
    left = keep // 2
    right = keep - left
    return s[:left] + "…" + s[-right:]


def _print_card_line(text: str):
    print(text)
    print("=" * len(text))


# box primitives


def _build_block(title, rows, cols=2, pad=1):
    """Generic 2D boxed table with N inner columns (ASCII)."""
    rows = [(k, v) for (k, v) in rows if v is not None and v != "-"]
    if not rows:
        rows = [("—", "—")]

    per_col = (len(rows) + cols - 1) // cols
    cols_data = [rows[i * per_col : (i + 1) * per_col] for i in range(cols)]

    widths = []
    for col in cols_data:
        k_w = max([len(str(k)) for k, _ in col] + [len(title)]) if col else len(title)
        v_w = max([len(_fmt(v)) for _, v in col] + [3]) if col else 3
        widths.append((k_w, v_w))

    def sep():
        return (" " * pad).join(
            "+-" + "-" * k + "-+-" + "-" * v + "-+" for (k, v) in widths
        )

    def head():
        parts = []
        for i, (k, v) in enumerate(widths):
            t = title if i == 0 else ""
            parts.append("| " + t.ljust(k) + " | " + "".ljust(v) + " |")
        return (" " * pad).join(parts)

    lines = [sep(), head(), sep()]
    h = max(len(c) for c in cols_data) if cols_data else 0
    for r in range(h):
        parts = []
        for i, (k_w, v_w) in enumerate(widths):
            col = cols_data[i]
            if r < len(col):
                k, v = col[r]
                parts.append(
                    "| " + str(k).ljust(k_w) + " | " + _fmt(v).ljust(v_w) + " |"
                )
            else:
                parts.append("| " + "".ljust(k_w) + " | " + "".ljust(v_w) + " |")
        lines.append((" " * pad).join(parts))
    lines.append(sep())
    return lines


def _side_by_side(a_lines, b_lines, gap=2):
    H = max(len(a_lines), len(b_lines))
    a = a_lines + [""] * (H - len(a_lines))
    b = b_lines + [""] * (H - len(b_lines))
    spacer = " " * gap
    return [a[i] + spacer + b[i] for i in range(H)]


def _box_artifact_metrics(path, created, finished, ppl, vloss, tloss, width_hint=100):
    """Single outer box: left=artifact, right=metrics, separated visually."""
    term_w = shutil.get_terminal_size().columns
    total_w = min(max(80, width_hint), term_w - 2)

    left_key_w = 12
    left_val_w = max(30, total_w // 2 - (left_key_w + 6))  # borders overhead
    right_key_w = 11
    right_val_w = 20

    p = _mid_trunc(path or "-", left_val_w + right_key_w + right_val_w + 6)
    ppl_s = "-" if ppl is None else _fmt(ppl)
    v_s = "-" if vloss is None else _fmt(vloss)
    t_s = "-" if tloss is None else _fmt(tloss)
    vt = f"{v_s}/{t_s}"

    # ASCII borders for compatibility
    top = (
        "+"
        + "-" * (left_key_w + left_val_w + 3)
        + "+"
        + "-" * (right_key_w + right_val_w + 3)
        + "+"
    )
    bot = (
        "+"
        + "-" * (left_key_w + left_val_w + 3)
        + "+"
        + "-" * (right_key_w + right_val_w + 3)
        + "+"
    )

    lines = [top]
    lines.append(
        "| "
        + "artifact".ljust(left_key_w + left_val_w + 1)
        + " | "
        + "metrics".ljust(right_key_w + right_val_w + 1)
        + " |"
    )
    lines.append(
        "+"
        + "-" * (left_key_w + left_val_w + 3)
        + "+"
        + "-" * (right_key_w + right_val_w + 3)
        + "+"
    )
    lines.append(
        "| "
        + "path".ljust(left_key_w)
        + " | "
        + p.ljust(left_val_w)
        + " | "
        + "".ljust(right_key_w)
        + " | "
        + "".ljust(right_val_w)
        + " |"
    )
    lines.append(
        "| "
        + "created_at".ljust(left_key_w)
        + " | "
        + _fmt(created).ljust(left_val_w)
        + " | "
        + "perplexity".ljust(right_key_w)
        + " | "
        + ppl_s.ljust(right_val_w)
        + " |"
    )
    lines.append(
        "| "
        + "finished_at".ljust(left_key_w)
        + " | "
        + _fmt(finished).ljust(left_val_w)
        + " | "
        + "val/train".ljust(right_key_w)
        + " | "
        + vt.ljust(right_val_w)
        + " |"
    )
    lines.append(bot)
    return lines


def _box_single_line(title, items, sep="  ||  "):
    """One-row rectangle used for tokenizer params."""
    pairs = [f"{k}: {_fmt(v)}" for (k, v) in items if v is not None]
    inner = sep.join(pairs) if pairs else "—"
    term_w = shutil.get_terminal_size().columns
    inner_max = min(len(inner), max(30, term_w - len(title) - 8))
    inner_shown = inner if len(inner) <= inner_max else _mid_trunc(inner, inner_max)

    top = "+" + "-" * (len(title) + 2) + " " + "-" * (len(inner_shown) + 2) + "+"
    head = "| " + title + " | " + inner_shown + " |"
    bot = "+" + "-" * (len(title) + 2) + " " + "-" * (len(inner_shown) + 2) + "+"
    return [top, head, bot]


# GPT list command


def list_gpt_models(root="."):
    reg_path = Path(root) / "experiments" / "models" / "gpt" / "registry.json"
    try:
        rows = json.loads(reg_path.read_text(encoding="utf-8"))
    except Exception:
        rows = []
    if not isinstance(rows, list) or not rows:
        print("No GPT finals saved yet.\n")
        return

    rows = sorted(rows, key=lambda r: r.get("perplexity", 1e9))
    print("\nGPT models\n")

    term_w = shutil.get_terminal_size().columns

    for idx, r in enumerate(rows):
        run_id = r.get("run_id", "unknown_run")
        path = r.get("path", "")
        created = r.get("created_at", "-")
        finished = r.get("finished_at", "-")
        ppl = r.get("perplexity")
        vloss = r.get("val_loss")
        tloss = r.get("train_loss")

        sig = r.get("signature", {})
        tok = sig.get("tokenizer", {})
        mdl = sig.get("model", {})
        trn = sig.get("train", {})
        opt = sig.get("optim", {})
        rtm = sig.get("runtime", {})

        # 0) header
        _print_card_line(f"[{idx}] {run_id}")

        # 1) artifact + metrics
        for line in _box_artifact_metrics(
            path, created, finished, ppl, vloss, tloss, width_hint=term_w - 6
        ):
            print(line)

        # 2) tokenizer (single-row)
        tok_items = [
            ("k", tok.get("k")),
            ("vocab", tok.get("vocab_size")),
            ("unk", tok.get("unk_id")),
        ]
        for line in _box_single_line("tokenizer params", tok_items):
            print(line)

        # 3) model/optimization params (merged in one block)
        rows_mo = [
            ("embd_dim", mdl.get("embd_dim")),
            ("n_layer", mdl.get("n_layer")),
            ("n_heads", mdl.get("n_heads")),
            ("dropout", mdl.get("dropout")),
            ("block_size", mdl.get("block_size")),
            ("learning_rate", opt.get("learning_rate")),
            ("weight_decay", opt.get("weight_decay")),
        ]
        dev = rtm.get("device", None)
        if dev in {"cpu", "cuda"}:
            rows_mo.append(("device", dev))
        if rtm.get("cuda") is False:  # hide when True
            rows_mo.append(("cuda", False))

        mo_block = _build_block("model/optimization params", rows_mo, cols=3)

        # 4) train block (with combined limits)
        tl, vl = trn.get("train_limit", None), trn.get("valid_limit", None)
        lim = None
        if (tl is not None) or (vl is not None):
            lim = f"{_fmt(tl)}/{_fmt(vl)}"
        rows_tr = [
            ("batch_size", trn.get("batch_size")),
            ("block_size", trn.get("block_size")),
            ("max_iters", trn.get("max_iters")),
            ("eval_interval", trn.get("eval_interval")),
            ("eval_iters", trn.get("eval_iters")),
            ("val/train lims", lim),
        ]
        tr_block = _build_block("train", rows_tr, cols=3)

        # 5) bottom shelf: [model/optim] | [train]  (stack if too wide)
        est_width = len(mo_block[0]) + 2 + len(tr_block[0])
        if est_width + 4 <= term_w:
            for line in _side_by_side(mo_block, tr_block, gap=2):
                print(line)
        else:
            for line in mo_block:
                print(line)
            for line in tr_block:
                print(line)

        print()
