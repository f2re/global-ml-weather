"""Build tables and SVG charts from completed pipeline artifacts, without inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
from html import escape
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable


MAX_JSON_BYTES = 32 * 1024**2
EPOCH_FIELDS = (
    "epoch",
    "training_horizon_hours",
    "validation_horizon_hours",
    "train_loss",
    "validation_loss",
    "elapsed_seconds",
)
SCORE_FIELDS = (
    "lead_hours",
    "variable",
    "units",
    "pressure_hpa",
    "count",
    "rmse",
    "mae",
    "bias",
    "control_rmse",
    "rmse_skill",
)


def _safe_path(path: str | Path) -> Path:
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Символические ссылки в пути запрещены.")
    return path


def sha256(path: str | Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def _pairs(items: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"Повторный ключ JSON: {key}")
        result[key] = value
    return result


def _finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Неконечное число в исходном отчёте.")
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    if isinstance(value, list):
        for item in value:
            _finite(item)


def read_json(path: str | Path) -> Any:
    path = _safe_path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("Требуется ограниченный обычный JSON-файл.")

    def invalid(value: str) -> None:
        raise ValueError(f"Неконечное число JSON: {value}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_pairs,
        parse_constant=invalid,
    )
    _finite(value)
    return value


def _number(value: Any, *, minimum: float | None = None) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Показатель должен быть конечным числом.")
    if minimum is not None and value < minimum:
        raise ValueError("Показатель меньше допустимого значения.")
    return value


def _integer(value: Any, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError("Требуется целое число допустимого диапазона.")
    return value


def _fingerprint(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("Отсутствует достоверно оформленный SHA256.")
    return value


def _artifact(root: Path, ref: Any) -> Path:
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ValueError("Артефакт требует path и sha256.")
    relative = ref["path"]
    if (
        not isinstance(relative, str)
        or not relative
        or ":" in relative
        or "\\" in relative
        or Path(relative).is_absolute()
        or any(p in ("", ".", "..") for p in relative.split("/"))
    ):
        raise ValueError("Недопустимый путь артефакта.")
    path = root
    for part in relative.split("/"):
        path = path / part
        if path.is_symlink():
            raise ValueError("Символические ссылки в артефактах запрещены.")
    if not path.is_file() or sha256(path) != _fingerprint(ref["sha256"]):
        raise ValueError("Артефакт отсутствует или его SHA256 изменён.")
    return path


def _csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def _svg(
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    series: list[tuple[str, list[tuple[float, float]]]],
    caption: str,
) -> None:
    points = [point for _, values in series for point in values]
    xs, ys = zip(*points)
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    if xmin == xmax:
        xmin, xmax = xmin - 0.5, xmax + 0.5
    if ymin == ymax:
        ymin, ymax = ymin - 0.5, ymax + 0.5
    pad = (ymax - ymin) * 0.05
    ymin, ymax = ymin - pad, ymax + pad

    def position(x: float, y: float) -> tuple[float, float]:
        return 90 + 640 * (x - xmin) / (xmax - xmin), 350 - 250 * (y - ymin) / (
            ymax - ymin
        )

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="820" height="480" viewBox="0 0 820 480">',
        '<rect width="820" height="480" fill="white"/>',
        '<g font-family="sans-serif" font-size="12" fill="#222">',
        f'<text x="30" y="25">{escape(title)}</text>',
        '<path d="M90 90V350H730" fill="none" stroke="#333"/>',
        f'<text x="360" y="395">{escape(xlabel)}</text>',
        f'<text x="15" y="65">{escape(ylabel)}</text>',
        f'<text x="30" y="440">{escape(caption)}</text>',
    ]
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        x = xmin + fraction * (xmax - xmin)
        y = ymin + fraction * (ymax - ymin)
        px, _ = position(x, ymin)
        _, py = position(xmin, y)
        parts.extend(
            [
                f'<text x="{px:.2f}" y="370">{x:.4g}</text>',
                f'<text x="30" y="{py:.2f}">{y:.4g}</text>',
            ]
        )
    colors = ("#165c9c", "#ba451c", "#347442")
    for index, (label, values) in enumerate(series):
        color = colors[index % len(colors)]
        coords = " ".join(
            f"{px:.2f},{py:.2f}" for px, py in (position(x, y) for x, y in values)
        )
        parts.append(
            f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        for x, y in values:
            px, py = position(x, y)
            parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3" fill="{color}"/>')
        parts.append(
            f'<text x="{90 + index * 220}" y="415" fill="{color}">{escape(label)}</text>'
        )
    parts.append("</g></svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def build_report(
    run: str | Path, output: str | Path, *, evaluation: str | Path | None = None
) -> dict[str, Any]:
    """Require a completed pipeline run and preserve every available score row."""
    run, output = _safe_path(run), _safe_path(output)
    if run.is_symlink() or not run.is_dir():
        raise ValueError("Требуется обычный каталог завершённого эксперимента.")
    sources = {}

    def load(path: Path) -> Any:
        value = read_json(path)
        sources[str(path)] = sha256(path)
        return value

    setup, history, trained, best = (
        load(run / name)
        for name in ("setup.json", "history.json", "report.json", "best.json")
    )
    if not all(isinstance(x, dict) for x in (setup, trained, best)):
        raise ValueError("Паспорта должны быть объектами JSON.")
    kind = trained.get("data_kind")
    if (
        kind not in ("synthetic", "real")
        or setup.get("data_kind") != kind
        or best.get("data_kind") != kind
    ):
        raise ValueError("Неизвестный или противоречивый тип данных.")
    if (
        trained.get("schema") != "weather-training-report-1"
        or trained.get("status") != "trained_research"
    ):
        raise ValueError("Обучение не имеет завершённого исследовательского отчёта.")
    if (
        trained.get("test_set_used_for_selection") is not False
        or best.get("selection") != "validation_only"
    ):
        raise ValueError("Не подтверждён выбор эпохи только по validation.")
    dataset = _fingerprint(trained.get("dataset_fingerprint"))
    normalization = _fingerprint(trained.get("normalization_fingerprint"))
    if any(x.get("dataset_fingerprint") != dataset for x in (setup, best)):
        raise ValueError("Отпечатки обучающей выборки расходятся.")
    for key in ("config", "software", "runtime"):
        if not isinstance(setup.get(key), dict) or not isinstance(best.get(key), dict):
            raise ValueError("Отсутствует конфигурация или среда выбранных весов.")
        left, right = setup[key], best[key]
        if key == "config":
            # Resuming may extend epochs; all model and numerical settings remain fixed.
            left = {k: v for k, v in left.items() if k != "epochs"}
            right = {k: v for k, v in right.items() if k != "epochs"}
        if left != right:
            raise ValueError("Конфигурация или среда расходится с выбранными весами.")
    if (
        not isinstance(history, list)
        or not history
        or history != trained.get("history")
    ):
        raise ValueError("История отсутствует или расходится с отчётом.")
    if trained.get("epochs_completed") != len(history):
        raise ValueError("Число завершённых эпох расходится с историей.")
    for index, row in enumerate(history, 1):
        if not isinstance(row, dict) or not set(EPOCH_FIELDS).issubset(row):
            raise ValueError("Неполная запись эпохи.")
        if _integer(row["epoch"], 1) != index:
            raise ValueError("Повторная или пропущенная эпоха.")
        for key in ("train_loss", "validation_loss", "elapsed_seconds"):
            _number(row[key], minimum=0)
        for key in ("training_horizon_hours", "validation_horizon_hours"):
            _integer(row[key], 1)
    if len({r["validation_horizon_hours"] for r in history}) != 1:
        raise ValueError("Горизонт validation изменялся между эпохами.")
    chosen = min(history, key=lambda r: r["validation_loss"])
    if (
        best.get("best_score") != chosen["validation_loss"]
        or trained.get("best_validation_loss") != chosen["validation_loss"]
    ):
        raise ValueError("Лучшая оценка расходится с историей.")
    weight_path = _artifact(run, best.get("weights"))
    if best["weights"]["path"] != f"epochs/{chosen['epoch']:06d}/weights.pt":
        raise ValueError("Контрольная точка не соответствует выбранной эпохе.")
    sources[str(weight_path)] = best["weights"]["sha256"]
    scores, assessed = [], None
    if evaluation is not None:
        assessed = load(Path(evaluation).absolute())
        if not isinstance(assessed, dict) or assessed.get("split") not in (
            "validation",
            "test",
        ):
            raise ValueError("Оценка требует validation или test.")
        if (
            assessed.get("data_kind") != kind
            or assessed.get("normalization_fingerprint") != normalization
            or assessed.get("selection_dataset_fingerprint") != dataset
            or assessed.get("weights") != best["weights"]
        ):
            raise ValueError("Оценка относится к другим данным, нормам или весам.")
        _fingerprint(assessed.get("dataset_fingerprint"))
        _number(assessed.get("normalized_forecast_loss"), minimum=0)
        if (
            not isinstance(assessed.get("sample_ids"), list)
            or not assessed["sample_ids"]
        ):
            raise ValueError("Отсутствуют идентификаторы оцениваемых примеров.")
        if not isinstance(assessed.get("scores"), list) or not assessed["scores"]:
            raise ValueError("Нет детальных показателей оценки.")
        identities: set[tuple[int, str, float | None]] = set()
        units: dict[tuple[str, float | None], str] = {}
        for row in assessed["scores"]:
            if not isinstance(row, dict) or not set(SCORE_FIELDS).issubset(row):
                raise ValueError("Неполная запись метрики.")
            _integer(row["lead_hours"], 1)
            _integer(row["count"], 1)
            if row["lead_hours"] > history[0]["validation_horizon_hours"]:
                raise ValueError("Срок метрики превышает обученный горизонт.")
            if row["pressure_hpa"] is not None:
                _number(row["pressure_hpa"], minimum=1)
            for key in ("variable", "units"):
                if not isinstance(row[key], str) or not row[key].strip():
                    raise ValueError("Отсутствует величина или единица.")
            identity = (row["lead_hours"], row["variable"], row["pressure_hpa"])
            if identity in identities:
                raise ValueError("Повторная запись метрики.")
            identities.add(identity)
            group = (row["variable"], row["pressure_hpa"])
            if group in units and units[group] != row["units"]:
                raise ValueError("Единица изменилась между сроками.")
            units[group] = row["units"]
            for key in ("rmse", "mae"):
                _number(row[key], minimum=0)
            _number(row["bias"])
            if row["mae"] > row["rmse"] + 1e-9 or abs(row["bias"]) > row["mae"] + 1e-9:
                raise ValueError("MAE, RMSE и смещение противоречат друг другу.")
            control = row["control_rmse"]
            skill = row["rmse_skill"]
            if control is not None:
                _number(control, minimum=0)
            expected = (
                None if control is None or control == 0 else 1 - row["rmse"] / control
            )
            if (
                expected is None
                and skill is not None
                or expected is not None
                and (
                    skill is None
                    or not math.isclose(
                        _number(skill), expected, rel_tol=1e-8, abs_tol=1e-10
                    )
                )
            ):
                raise ValueError("Показатель skill не соответствует контрольному RMSE.")
            scores.append({key: row[key] for key in SCORE_FIELDS})
    if output.exists() or output.is_symlink():
        raise FileExistsError("Отчёт не перезаписывается.")
    output.mkdir(parents=True)
    _csv(output / "epochs.csv", EPOCH_FIELDS, history)
    label = "synthetic" if kind == "synthetic" else "real (source claims not certified)"
    caption = f"{label}; маски и период: см. исходные паспорта; источник: epochs.csv"
    _svg(
        output / "learning.svg",
        "Функция потерь по эпохам",
        "Эпоха",
        "Нормализованная функция потерь",
        [
            (key, [(r["epoch"], r[key]) for r in history])
            for key in ("train_loss", "validation_loss")
        ],
        caption,
    )
    _svg(
        output / "epoch_time.svg",
        "Измеренное время эпохи",
        "Эпоха",
        "Время, с",
        [("elapsed_seconds", [(r["epoch"], r["elapsed_seconds"]) for r in history])],
        caption,
    )
    if scores:
        assert assessed is not None
        _csv(output / "scores.csv", SCORE_FIELDS, scores)
        groups = sorted(
            {(r["variable"], r["units"], r["pressure_hpa"]) for r in scores}, key=str
        )
        for index, (variable, unit, pressure) in enumerate(groups):
            rows = sorted(
                [
                    r
                    for r in scores
                    if (r["variable"], r["units"], r["pressure_hpa"])
                    == (variable, unit, pressure)
                ],
                key=lambda r: r["lead_hours"],
            )
            title = (
                f"{variable}; {pressure} гПа"
                if pressure is not None
                else f"{variable}; поверхность"
            )
            _svg(
                output / f"lead_{index:03d}.svg",
                title,
                "Заблаговременность, ч",
                f"Ошибка, {unit}",
                [
                    (key, [(r["lead_hours"], r[key]) for r in rows])
                    for key in ("rmse", "mae", "bias")
                ],
                f'{label}; {assessed["split"]}; исходные целевые маски; источник: scores.csv',
            )
    summary = {
        "schema": "weather-experiment-summary-1",
        "status": "artifact_summary",
        "data_kind": kind,
        "meteorologically_validated": False,
        "best_epoch": chosen["epoch"],
        "history": history,
        "scores": scores,
        "dataset_fingerprint": dataset,
        "normalization_fingerprint": normalization,
        "weights": best["weights"],
        "config": best["config"],
        "software": best["software"],
        "runtime": best["runtime"],
        "measured_resources": {
            "epoch_elapsed_seconds": [r["elapsed_seconds"] for r in history]
        },
        "estimated_resources": {
            "activation_bytes": setup.get("estimated_activation_bytes")
        },
        "evaluation": (
            {k: v for k, v in assessed.items() if k != "scores"} if assessed else None
        ),
        "source_sha256": sources,
        "limitations": [
            "Исходные нормы и массивы здесь не перечитываются; их отпечатки взяты из обучения.",
            "Память, FLOPs, покрытие, периоды и независимость не выводятся из числа целей.",
            "Потери train и validation имеют разный состав и могут иметь разные горизонты.",
            "Отсутствующие сроки не интерполируются; сравнение архитектур не выполняется.",
        ],
    }
    _json(output / "summary.json", summary)
    _json(
        output / "artifacts.json",
        {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = build_report(args.run, args.output, evaluation=args.evaluation)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"Отчёт не построен: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "data_kind": result["data_kind"],
                "output": str(args.output),
                "best_epoch": result["best_epoch"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
