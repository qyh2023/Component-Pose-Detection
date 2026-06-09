#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from mask import MicrolensMaskDemo, angle_from_y, imread_unicode, imwrite_unicode, rect_contour


def angle_distance(a: float, b: float) -> float:
    delta = abs(a - b) % 180.0
    return min(delta, 180.0 - delta)


def intersection_fraction(a: tuple, b: tuple) -> float:
    status, points = cv2.rotatedRectangleIntersection(a, b)
    if status == cv2.INTERSECT_NONE or points is None:
        return 0.0
    area = abs(float(cv2.contourArea(points)))
    area_a = max(1.0, float(a[1][0] * a[1][1]))
    area_b = max(1.0, float(b[1][0] * b[1][1]))
    return area / min(area_a, area_b)


class RectSetSolver:
    def __init__(
        self,
        detector: MicrolensMaskDemo,
        box_scale: float = 0.93,
        angle_step: int = 5,
        mask_weight: float = 2.2,
        edge_weight: float = 1.4,
        kmeans_weight: float = 1.1,
        kmeans_sigma_factor: float = 0.55,
    ) -> None:
        self.detector = detector
        self.box_scale = float(box_scale)
        self.angle_step = max(3, int(angle_step))
        self.mask_weight = float(mask_weight)
        self.edge_weight = float(edge_weight)
        self.kmeans_weight = float(kmeans_weight)
        self.kmeans_sigma_factor = max(0.1, float(kmeans_sigma_factor))

    @staticmethod
    def component_stats(mask: np.ndarray) -> list[dict]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [MicrolensMaskDemo.contour_stats(contour) for contour in contours]

    def estimate_template(self, components: list[dict]) -> tuple[float, float, float]:
        plausible = [
            item
            for item in components
            if self.detector.min_area <= item["area"] <= self.detector.max_area
            and self.detector.lying_ar_min <= item["aspect"] <= 2.8
            and item["extent"] >= 0.35
        ]
        if len(plausible) < 5:
            plausible = sorted(components, key=lambda item: item["area"])[: max(3, len(components) // 2)]
        single_area = float(np.median([item["area"] for item in plausible]))
        long_side = float(np.median([item["long"] for item in plausible]))
        short_side = float(np.median([item["short"] for item in plausible]))
        return single_area, long_side, short_side

    def candidate_score(
        self,
        crop_mask: np.ndarray,
        crop_edges: np.ndarray,
        rect: tuple,
        kmeans_centers: np.ndarray | None = None,
        short_side: float | None = None,
    ) -> tuple[float, float, float, float]:
        filled = np.zeros_like(crop_mask)
        outline = np.zeros_like(crop_mask)
        box = np.intp(cv2.boxPoints(rect)).reshape(-1, 1, 2)
        cv2.drawContours(filled, [box], -1, 255, cv2.FILLED)
        cv2.drawContours(outline, [box], -1, 255, 1)
        template_area = max(1, cv2.countNonZero(filled))
        inside = cv2.countNonZero(cv2.bitwise_and(filled, crop_mask)) / template_area
        edge_support = cv2.dilate(crop_edges, np.ones((3, 3), np.uint8))
        perimeter = max(1, cv2.countNonZero(outline))
        edge_alignment = cv2.countNonZero(cv2.bitwise_and(outline, edge_support)) / perimeter
        kmeans_support = 0.0
        if kmeans_centers is not None and len(kmeans_centers):
            center = np.array(rect[0], dtype=np.float32)
            distance = float(np.min(np.linalg.norm(kmeans_centers - center, axis=1)))
            sigma = max(1.0, self.kmeans_sigma_factor * float(short_side or 1.0))
            kmeans_support = math.exp(-0.5 * (distance / sigma) ** 2)
        score = (
            self.mask_weight * inside
            + self.edge_weight * edge_alignment
            + self.kmeans_weight * kmeans_support
        )
        return float(score), float(inside), float(edge_alignment), float(kmeans_support)

    @staticmethod
    def kmeans_centers(component_mask: np.ndarray, count: int) -> np.ndarray:
        ys, xs = np.where(component_mask > 0)
        if count <= 0 or len(xs) < count:
            return np.empty((0, 2), dtype=np.float32)
        coords = np.column_stack([xs, ys]).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        cv2.setRNGSeed(20260609)
        _compactness, _labels, centers = cv2.kmeans(
            coords,
            count,
            None,
            criteria,
            12,
            cv2.KMEANS_PP_CENTERS,
        )
        return centers.astype(np.float32)

    def generate_candidates(
        self,
        component_mask: np.ndarray,
        component_edges: np.ndarray,
        count: int,
        long_side: float,
        short_side: float,
        kmeans_centers: np.ndarray,
    ) -> list[dict]:
        h, w = component_mask.shape
        object_size = (long_side * self.box_scale, short_side * self.box_scale)
        candidates: list[dict] = []
        mask_float = component_mask.astype(np.float32) / 255.0
        edge_float = cv2.dilate(component_edges, np.ones((2, 2), np.uint8)).astype(np.float32) / 255.0

        template_size = int(math.ceil(math.hypot(*object_size))) + 10
        if template_size % 2 == 0:
            template_size += 1
        half = template_size // 2
        padded_mask = cv2.copyMakeBorder(mask_float, half, half, half, half, cv2.BORDER_CONSTANT)
        padded_edges = cv2.copyMakeBorder(edge_float, half, half, half, half, cv2.BORDER_CONSTANT)
        yy, xx = np.indices((h, w), dtype=np.float32)
        sigma = max(1.0, self.kmeans_sigma_factor * short_side)
        if len(kmeans_centers):
            distance_sq = np.min(
                (xx[..., None] - kmeans_centers[:, 0]) ** 2
                + (yy[..., None] - kmeans_centers[:, 1]) ** 2,
                axis=2,
            )
            kmeans_map = np.exp(-0.5 * distance_sq / (sigma * sigma))
        else:
            kmeans_map = np.zeros((h, w), dtype=np.float32)

        for angle in range(0, 180, self.angle_step):
            filled_template = np.zeros((template_size, template_size), dtype=np.uint8)
            edge_template = np.zeros_like(filled_template)
            box = rect_contour((float(half), float(half)), object_size, float(angle))
            cv2.drawContours(filled_template, [box], -1, 255, cv2.FILLED)
            cv2.drawContours(edge_template, [box], -1, 255, 1)
            filled_float = filled_template.astype(np.float32) / 255.0
            edge_template_float = edge_template.astype(np.float32) / 255.0
            area = max(1.0, float(np.sum(filled_float)))
            perimeter = max(1.0, float(np.sum(edge_template_float)))
            inside = cv2.matchTemplate(padded_mask, filled_float, cv2.TM_CCORR) / area
            edge_match = cv2.matchTemplate(padded_edges, edge_template_float, cv2.TM_CCORR) / perimeter
            response = (
                self.mask_weight * inside
                + self.edge_weight * edge_match
                + self.kmeans_weight * kmeans_map
            )
            response = response[:h, :w]
            local_max = cv2.dilate(response, np.ones((7, 7), np.float32))
            ys, xs = np.where((inside[:h, :w] >= 0.58) & (response == local_max))
            if len(xs) > max(8, count * 3):
                order = np.argsort(response[ys, xs])[::-1][: max(8, count * 3)]
                ys = ys[order]
                xs = xs[order]
            for cy, cx in zip(ys, xs):
                rect = ((float(cx), float(cy)), object_size, float(angle))
                candidates.append(
                    {
                        "rect": rect,
                        "center": np.array([float(cx), float(cy)], dtype=np.float32),
                        "score": float(response[cy, cx]),
                        "inside": float(inside[cy, cx]),
                        "edge": float(edge_match[cy, cx]),
                        "kmeans": float(kmeans_map[cy, cx]),
                        "source": "template",
                    }
                )

        # Ensure every K-means center contributes angle hypotheses even when
        # it is not a local maximum of the sliding-template response.
        for center in kmeans_centers:
            cx, cy = map(float, center)
            for angle in range(0, 180, self.angle_step):
                rect = ((cx, cy), object_size, float(angle))
                score, inside_value, edge_value, kmeans_value = self.candidate_score(
                    component_mask,
                    component_edges,
                    rect,
                    kmeans_centers,
                    short_side,
                )
                if inside_value >= 0.50:
                    candidates.append(
                        {
                            "rect": rect,
                            "center": center.copy(),
                            "score": score,
                            "inside": inside_value,
                            "edge": edge_value,
                            "kmeans": kmeans_value,
                            "source": "kmeans_center",
                        }
                    )

        candidates.sort(key=lambda item: item["score"], reverse=True)
        deduped: list[dict] = []
        for candidate in candidates:
            duplicate = any(
                np.linalg.norm(candidate["center"] - kept["center"]) < 2.5
                and angle_distance(candidate["rect"][2], kept["rect"][2]) < self.angle_step * 1.5
                for kept in deduped
            )
            if not duplicate:
                deduped.append(candidate)
            if len(deduped) >= min(260, max(50, count * 18)):
                break
        return deduped

    @staticmethod
    def conflicts(
        a: dict,
        b: dict,
        short_side: float,
        center_factor: float = 0.52,
        overlap_limit: float = 0.36,
    ) -> bool:
        center_distance = float(np.linalg.norm(a["center"] - b["center"]))
        if center_distance < center_factor * short_side:
            return True
        # The fitted rectangles include a small display margin, so touching
        # physical objects may have moderately overlapping display boxes.
        return intersection_fraction(a["rect"], b["rect"]) > overlap_limit

    def solve_candidates(self, candidates: list[dict], count: int, short_side: float) -> list[dict]:
        if len(candidates) < count:
            return []
        n = len(candidates)
        scores = np.array([item["score"] for item in candidates], dtype=np.float64)
        conflict_levels = ((0.68, 0.08), (0.58, 0.16))
        for center_factor, overlap_limit in conflict_levels:
            conflict_pairs = []
            for i in range(n):
                for j in range(i + 1, n):
                    if self.conflicts(
                        candidates[i],
                        candidates[j],
                        short_side,
                        center_factor,
                        overlap_limit,
                    ):
                        conflict_pairs.append((i, j))

            matrix = lil_matrix((1 + len(conflict_pairs), n), dtype=np.float64)
            matrix[0, :] = 1.0
            lower = np.full(1 + len(conflict_pairs), -np.inf, dtype=np.float64)
            upper = np.ones(1 + len(conflict_pairs), dtype=np.float64)
            lower[0] = count
            upper[0] = count
            for row, (i, j) in enumerate(conflict_pairs, start=1):
                matrix[row, i] = 1.0
                matrix[row, j] = 1.0

            result = milp(
                c=-scores,
                integrality=np.ones(n),
                bounds=Bounds(np.zeros(n), np.ones(n)),
                constraints=LinearConstraint(matrix.tocsr(), lower, upper),
                options={"time_limit": 8.0, "mip_rel_gap": 0.02},
            )
            if result.x is not None:
                selected = [candidates[i] for i, value in enumerate(result.x) if value > 0.5]
                if len(selected) == count:
                    return selected

        selected: list[dict] = []
        for candidate in candidates:
            if all(
                not self.conflicts(candidate, kept, short_side, 0.54, 0.16)
                for kept in selected
            ):
                selected.append(candidate)
                if len(selected) == count:
                    return selected
        return []

    def solve_component(
        self,
        full_mask: np.ndarray,
        edges: np.ndarray,
        item: dict,
        count: int,
        long_side: float,
        short_side: float,
    ) -> tuple[list[np.ndarray], dict]:
        x, y, w, h = cv2.boundingRect(item["contour"])
        pad = int(math.ceil(long_side))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(full_mask.shape[1], x + w + pad)
        y1 = min(full_mask.shape[0], y + h + pad)
        component = np.zeros_like(full_mask)
        cv2.drawContours(component, [item["contour"]], -1, 255, cv2.FILLED)
        crop_mask = component[y0:y1, x0:x1]
        crop_edges = cv2.bitwise_and(edges[y0:y1, x0:x1], cv2.dilate(crop_mask, np.ones((3, 3), np.uint8)))
        centers = self.kmeans_centers(crop_mask, count)
        candidates = self.generate_candidates(
            crop_mask,
            crop_edges,
            count,
            long_side,
            short_side,
            centers,
        )
        selected = self.solve_candidates(candidates, count, short_side)
        boxes = []
        for candidate in selected:
            (cx, cy), size, angle = candidate["rect"]
            box = rect_contour((cx + x0, cy + y0), size, angle)
            boxes.append(box)
        return boxes, {
            "requested_count": int(count),
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "mean_inside": round(float(np.mean([x["inside"] for x in selected])), 3) if selected else 0.0,
            "mean_edge": round(float(np.mean([x["edge"] for x in selected])), 3) if selected else 0.0,
            "mean_kmeans": round(float(np.mean([x["kmeans"] for x in selected])), 3) if selected else 0.0,
            "kmeans_center_count": len(centers),
            "selected_from_kmeans": sum(x["source"] == "kmeans_center" for x in selected),
            "method": "rectangle_set_milp" if selected else "fallback_single",
        }

    def recognize(
        self,
        tray_crop: np.ndarray,
        mask: np.ndarray,
        edges: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
        components = [
            item
            for item in self.component_stats(mask)
            if self.detector.min_area <= item["area"] <= self.detector.max_area * 12
            and item["extent"] >= 0.18
            and item["aspect"] <= 4.0
        ]
        if not components:
            return tray_crop.copy(), np.zeros_like(mask), [], {}

        single_area, long_side, short_side = self.estimate_template(components)
        annotated = tray_crop.copy()
        rect_mask = np.zeros_like(mask)
        records = []
        split_info = []

        for item in sorted(components, key=lambda value: (value["center"][1], value["center"][0])):
            count = int(np.clip(round(item["area"] / single_area), 1, 24))
            pieces: list[np.ndarray] = []
            if count >= 2:
                pieces, info = self.solve_component(mask, edges, item, count, long_side, short_side)
                info.update(
                    {
                        "component_area": round(item["area"], 3),
                        "area_ratio": round(item["area"] / single_area, 3),
                        "center_x": round(item["center"][0], 3),
                        "center_y": round(item["center"][1], 3),
                    }
                )
                split_info.append(info)
            if not pieces:
                (cx, cy), (w, h), angle = item["rect"]
                if item["aspect"] < self.detector.lying_ar_min and item["area"] < 1.25 * single_area:
                    size = (short_side * self.box_scale, short_side * self.box_scale)
                elif w >= h:
                    size = (long_side * self.box_scale, short_side * self.box_scale)
                else:
                    size = (short_side * self.box_scale, long_side * self.box_scale)
                pieces = [rect_contour((float(cx), float(cy)), size, float(angle))]
            for piece in pieces:
                stats = MicrolensMaskDemo.contour_stats(piece)
                pose = "lying" if stats["aspect"] >= self.detector.lying_ar_min else "standing"
                color = (40, 190, 40) if pose == "lying" else (0, 150, 255)
                cv2.drawContours(rect_mask, [piece], -1, 255, cv2.FILLED)
                cv2.drawContours(annotated, [piece], -1, color, 1)
                cx, cy = stats["center"]
                cv2.circle(annotated, (round(cx), round(cy)), 2, (0, 0, 255), -1)
                angle = angle_from_y(stats["rect"])
                if pose == "lying":
                    radians = math.radians(angle)
                    end = (round(cx + 19 * math.sin(radians)), round(cy + 19 * math.cos(radians)))
                    cv2.arrowedLine(annotated, (round(cx), round(cy)), end, (255, 0, 0), 1, tipLength=0.22)
                sample_id = len(records) + 1
                cv2.putText(
                    annotated,
                    f"{sample_id}:{'L' if pose == 'lying' else 'S'}",
                    (round(cx) + 4, round(cy) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    color,
                    1,
                    cv2.LINE_AA,
                )
                records.append(
                    {
                        "sample_id": sample_id,
                        "pose": pose,
                        "center_x": round(cx, 3),
                        "center_y": round(cy, 3),
                        "angle_deg": angle if pose == "lying" else "",
                        "area_px": round(stats["area"], 3),
                        "width_px": round(stats["width"], 3),
                        "height_px": round(stats["height"], 3),
                        "aspect_ratio": round(stats["aspect"], 3),
                    }
                )

        lying_count = sum(item["pose"] == "lying" for item in records)
        standing_count = sum(item["pose"] == "standing" for item in records)
        cv2.putText(
            annotated,
            f"lying={lying_count}, standing={standing_count}",
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (40, 190, 40),
            2,
            cv2.LINE_AA,
        )
        info = {
            "single_mask_area": round(single_area, 3),
            "template_long": round(long_side * self.box_scale, 3),
            "template_short": round(short_side * self.box_scale, 3),
            "component_splits": split_info,
        }
        return annotated, rect_mask, records, info


def process_image(
    image_path: Path,
    output_dir: Path,
    max_side: int,
    box_scale: float,
    angle_step: int,
    mask_weight: float,
    edge_weight: float,
    kmeans_weight: float,
    kmeans_sigma_factor: float,
) -> dict:
    detector = MicrolensMaskDemo(box_scale=0.98)
    image = imread_unicode(image_path)
    scale = min(1.0, max_side / float(max(image.shape[:2])))
    if scale < 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    tray_crop, tray_mask, tray_bbox = detector.find_tray_crop(image)
    mask, debug, mask_info = detector.build_mask(tray_crop, tray_mask)
    solver = RectSetSolver(
        detector,
        box_scale=box_scale,
        angle_step=angle_step,
        mask_weight=mask_weight,
        edge_weight=edge_weight,
        kmeans_weight=kmeans_weight,
        kmeans_sigma_factor=kmeans_sigma_factor,
    )
    annotated, rect_mask, records, solver_info = solver.recognize(tray_crop, mask, debug["edges"])

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    imwrite_unicode(output_dir / f"{stem}_00_tray_crop.png", tray_crop)
    imwrite_unicode(output_dir / f"{stem}_02_mask.png", mask)
    imwrite_unicode(output_dir / f"{stem}_02a_edges.png", debug["edges"])
    imwrite_unicode(output_dir / f"{stem}_04_rect_mask.png", rect_mask)
    imwrite_unicode(output_dir / f"{stem}_05_annotated.png", annotated)

    csv_path = output_dir / f"{stem}_result.csv"
    if records:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    result = {
        "image": str(image_path),
        "tray_bbox": tray_bbox,
        "process_scale": round(scale, 4),
        "lying_count": sum(item["pose"] == "lying" for item in records),
        "standing_count": sum(item["pose"] == "standing" for item in records),
        "mask_info": mask_info,
        "solver_info": solver_info,
        "detections": records,
    }
    (output_dir / f"{stem}_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Non-overlapping rectangle solver using the mask.py preprocessing")
    parser.add_argument("image", nargs="?", default="Image_20260515110818495.bmp")
    parser.add_argument("-o", "--output-dir", default="outputs_rect_solver")
    parser.add_argument("--max-side", type=int, default=1800)
    parser.add_argument("--box-scale", type=float, default=0.93)
    parser.add_argument("--angle-step", type=int, default=5)
    parser.add_argument("--mask-weight", type=float, default=2.2)
    parser.add_argument("--edge-weight", type=float, default=1.4)
    parser.add_argument("--kmeans-weight", type=float, default=1.1)
    parser.add_argument("--kmeans-sigma-factor", type=float, default=0.55)
    args = parser.parse_args()
    result = process_image(
        Path(args.image),
        Path(args.output_dir),
        max_side=args.max_side,
        box_scale=args.box_scale,
        angle_step=args.angle_step,
        mask_weight=args.mask_weight,
        edge_weight=args.edge_weight,
        kmeans_weight=args.kmeans_weight,
        kmeans_sigma_factor=args.kmeans_sigma_factor,
    )
    print(f"lying_count: {result['lying_count']}")
    print(f"standing_count: {result['standing_count']}")
    print(f"single_mask_area: {result['solver_info'].get('single_mask_area')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
