"""Configurable multichannel matching across detected objects."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import re
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from config.settings import MatchingConfig
from data_models.models import DetectedObject, OverlapGroupRecord, SectionChannelResult

CONCRETE_OVERLAP_TERM_PATTERN = re.compile(r"(CH\d+)\s*([+-]?)", re.IGNORECASE)
OVERLAP_SET_TERM_PATTERN = re.compile(r"(CH\d+)\s*(\+\-|\+|-|±)?", re.IGNORECASE)
FALLBACK_CHANNEL_PATTERN = re.compile(r"CH\d+", re.IGNORECASE)
SIZE_BONUS = 1_000_000_000.0


@dataclass(slots=True)
class OverlapSpec:
    """Parsed overlap-set specification such as ``CH2+CH3+CH1-``."""

    raw: str
    terms: tuple[tuple[str, bool], ...]

    @property
    def label(self) -> str:
        return "".join(f"{channel}{'+' if positive else '-'}" for channel, positive in self.terms)

    @property
    def positive_channels(self) -> tuple[str, ...]:
        return tuple(channel for channel, positive in self.terms if positive)

    @property
    def all_channels(self) -> tuple[str, ...]:
        return tuple(channel for channel, _ in self.terms)


@dataclass(slots=True)
class ChannelRule:
    """One explicit pairwise overlap rule between two channels."""

    channels: tuple[str, str]
    rule: dict[str, Any]


@dataclass(slots=True)
class PairCandidate:
    """One candidate edge between two channel-specific objects."""

    first_id: str
    second_id: str
    score: float


@dataclass(slots=True)
class GroupCandidate:
    """One candidate multi-channel biological-cell group."""

    channels: tuple[str, ...]
    object_ids: tuple[str, ...]
    score: float


def _channel_key_for_result(result: SectionChannelResult) -> str:
    return (result.bundle.image_channel or result.bundle.channel).upper()


def _channel_key_for_object(obj: DetectedObject) -> str:
    return (obj.image_channel or obj.channel).upper()


def _normalize_combo(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted((value.strip().upper() for value in values if value and value.strip()), key=str.upper))


def _combo_key(values: list[str] | tuple[str, ...]) -> str:
    return "+".join(_normalize_combo(values))


def parse_overlap_spec(spec: str) -> OverlapSpec:
    """Parse signed overlap specifications, defaulting missing signs to positive."""

    matches = CONCRETE_OVERLAP_TERM_PATTERN.findall(spec.upper())
    if not matches:
        fallback_channels = [token.upper() for token in FALLBACK_CHANNEL_PATTERN.findall(spec.upper())]
        matches = [(token, "+") for token in fallback_channels]

    terms: list[tuple[str, bool]] = []
    seen: set[str] = set()
    for channel, sign in matches:
        channel = channel.strip().upper()
        if not channel or channel in seen:
            continue
        terms.append((channel, sign != "-"))
        seen.add(channel)
    if not terms:
        raise ValueError(f"Could not parse overlap spec: {spec}")
    return OverlapSpec(raw=spec, terms=tuple(terms))


def expand_overlap_set_spec(spec: str) -> list[OverlapSpec]:
    """Expand a raw overlap-set spec, supporting ``+-`` / ``±`` branches."""

    matches = OVERLAP_SET_TERM_PATTERN.findall(spec.upper())
    if not matches:
        fallback_channels = [token.upper() for token in FALLBACK_CHANNEL_PATTERN.findall(spec.upper())]
        matches = [(token, "+") for token in fallback_channels]

    parsed_terms: list[tuple[str, tuple[bool, ...]]] = []
    seen: set[str] = set()
    for channel, raw_sign in matches:
        channel = channel.strip().upper()
        if not channel or channel in seen:
            continue
        sign = raw_sign or "+"
        if sign in {"+-", "±"}:
            options = (True, False)
        elif sign == "-":
            options = (False,)
        else:
            options = (True,)
        parsed_terms.append((channel, options))
        seen.add(channel)
    if not parsed_terms:
        raise ValueError(f"Could not parse overlap set spec: {spec}")

    expanded: list[OverlapSpec] = []

    def backtrack(index: int, current: list[tuple[str, bool]]) -> None:
        if index >= len(parsed_terms):
            expanded.append(OverlapSpec(raw=spec, terms=tuple(current)))
            return
        channel, options = parsed_terms[index]
        for option in options:
            current.append((channel, option))
            backtrack(index + 1, current)
            current.pop()

    backtrack(0, [])
    return expanded


def _expand_specs(config: MatchingConfig) -> list[OverlapSpec]:
    parsed: list[OverlapSpec] = []
    seen: set[str] = set()
    for spec in config.combinations:
        if not spec or not str(spec).strip():
            continue
        for expanded in expand_overlap_set_spec(str(spec)):
            if expanded.label in seen:
                continue
            seen.add(expanded.label)
            parsed.append(expanded)
    return parsed


def _positive_channel_sets(specs: list[OverlapSpec]) -> list[tuple[str, ...]]:
    """Return positive-channel sets that should be explicitly optimized."""

    relevant: dict[tuple[str, ...], tuple[str, ...]] = {}
    for spec in specs:
        positive = _normalize_combo(spec.positive_channels)
        if len(positive) >= 2:
            relevant[positive] = positive
    return list(relevant.values())


def _pair_channel_key(first_channel: str, second_channel: str) -> tuple[str, str]:
    return tuple(sorted((first_channel.upper(), second_channel.upper()), key=str.upper))


def _pair_rule(config: MatchingConfig, first_channel: str, second_channel: str) -> dict[str, Any]:
    pair_key = _combo_key((first_channel, second_channel))
    for raw_key, value in config.pair_rules.items():
        try:
            channels = _normalize_combo(parse_overlap_spec(str(raw_key)).all_channels)
        except ValueError:
            channels = _normalize_combo(re.findall(r"CH\d+", str(raw_key).upper()))
        if len(channels) == 2 and _combo_key(channels) == pair_key:
            rule = value
            break
    else:
        rule = {}
    return {
        "method": str(rule.get("method", config.method)),
        "distance_threshold_px": float(rule.get("distance_threshold_px", config.distance_threshold_px)),
        "fallback_distance_threshold_px": float(rule.get("fallback_distance_threshold_px", 0.0)),
        "iou_threshold": float(rule.get("iou_threshold", config.iou_threshold)),
        "overlap_threshold_px": int(rule.get("overlap_threshold_px", config.overlap_threshold_px)),
    }


def _available_rule_edges(config: MatchingConfig, channels: tuple[str, ...]) -> list[tuple[str, str]]:
    available: list[tuple[str, str]] = []
    for first_channel, second_channel in itertools.combinations(sorted(channels, key=str.upper), 2):
        raw_rule = _pair_rule(config, first_channel, second_channel)
        raw_pair = config.pair_rules.get(_combo_key((first_channel, second_channel)))
        if raw_pair is not None or any(
            _combo_key((first_channel, second_channel))
            == _combo_key(re.findall(r"CH\d+", str(key).upper()))
            for key in config.pair_rules
        ):
            available.append((first_channel, second_channel))
    return available


def _validate_required_pair_rules(config: MatchingConfig, specs: list[OverlapSpec]) -> None:
    normalized_pair_rules = {
        _combo_key(re.findall(r"CH\d+", str(key).upper())): value for key, value in config.pair_rules.items()
    }
    missing_descriptions: list[str] = []
    for spec in specs:
        channels = _normalize_combo(spec.all_channels)
        if len(channels) < 2:
            continue
        present_pairs = {
            tuple(sorted((first_channel, second_channel), key=str.upper))
            for first_channel, second_channel in itertools.combinations(channels, 2)
            if _combo_key((first_channel, second_channel)) in normalized_pair_rules
        }
        if not present_pairs:
            missing_descriptions.append(f"{spec.label}: no overlap rule for {', '.join(channels)}")
            continue
        adjacency: dict[str, set[str]] = defaultdict(set)
        for first_channel, second_channel in present_pairs:
            adjacency[first_channel].add(second_channel)
            adjacency[second_channel].add(first_channel)
        visited: set[str] = set()
        stack = [channels[0]]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(adjacency[current] - visited)
        if set(channels) - visited:
            missing_descriptions.append(
                f"{spec.label}: overlap rules are insufficient for channels {', '.join(channels)}"
            )
    if missing_descriptions:
        raise ValueError("Missing overlap rules:\n" + "\n".join(missing_descriptions))


def _bbox_overlap_metrics(a: DetectedObject, b: DetectedObject) -> tuple[int, float]:
    ax0, ay0, ax1, ay1 = a.bbox
    bx0, by0, bx1, by1 = b.bbox
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0, 0.0

    a_crop_x0 = ix0 - ax0
    a_crop_y0 = iy0 - ay0
    a_crop_x1 = ix1 - ax0
    a_crop_y1 = iy1 - ay0
    b_crop_x0 = ix0 - bx0
    b_crop_y0 = iy0 - by0
    b_crop_x1 = ix1 - bx0
    b_crop_y1 = iy1 - by0

    a_mask = a.mask_crop[a_crop_y0:a_crop_y1, a_crop_x0:a_crop_x1]
    b_mask = b.mask_crop[b_crop_y0:b_crop_y1, b_crop_x0:b_crop_x1]
    overlap = int(np.logical_and(a_mask, b_mask).sum())
    union = int(np.logical_or(a_mask, b_mask).sum())
    return overlap, (overlap / union) if union else 0.0


def _pair_score_for_candidate(
    *,
    rule: dict[str, Any],
    distance: float,
    overlap_px: int,
    iou: float,
) -> float | None:
    method = str(rule["method"])
    distance_threshold = float(rule["distance_threshold_px"])
    fallback_distance = float(rule["fallback_distance_threshold_px"])
    overlap_threshold = int(rule["overlap_threshold_px"])
    iou_threshold = float(rule["iou_threshold"])

    if method == "centroid_distance":
        if distance <= distance_threshold:
            return max(0.0, distance_threshold - distance) + 1_000.0
        return None
    if method == "mask_overlap":
        if overlap_px >= overlap_threshold:
            return 2_000_000.0 + float(overlap_px)
        return None
    if method in {"mask_overlap_then_centroid", "mask_overlap_or_distance"}:
        if overlap_px >= overlap_threshold:
            return 2_000_000.0 + float(overlap_px)
        if fallback_distance > 0 and distance <= fallback_distance:
            return max(0.0, fallback_distance - distance) + 1_000.0
        return None
    if iou >= iou_threshold:
        return 2_000_000.0 + (float(iou) * 1000.0)
    return None


def _pair_candidates(
    first: list[DetectedObject],
    second: list[DetectedObject],
    rule: dict[str, Any],
) -> list[PairCandidate]:
    if not first or not second:
        return []

    first_xy = np.array([[obj.centroid_x_px, obj.centroid_y_px] for obj in first], dtype=np.float32)
    second_xy = np.array([[obj.centroid_x_px, obj.centroid_y_px] for obj in second], dtype=np.float32)
    tree = cKDTree(second_xy)

    candidates_by_ids: dict[tuple[str, str], PairCandidate] = {}
    radius = max(
        float(rule["distance_threshold_px"]),
        float(rule["fallback_distance_threshold_px"]),
        float(rule["overlap_threshold_px"]),
        64.0 if str(rule["method"]) != "centroid_distance" else 1.0,
    )
    for index, point in enumerate(first_xy):
        distances, neighbor_indices = tree.query(
            point,
            k=min(16, len(second)),
            distance_upper_bound=radius,
        )
        for distance, neighbor_index in zip(np.atleast_1d(distances), np.atleast_1d(neighbor_indices), strict=False):
            if np.isinf(distance) or int(neighbor_index) >= len(second):
                continue
            first_obj = first[index]
            second_obj = second[int(neighbor_index)]
            overlap_px, iou = _bbox_overlap_metrics(first_obj, second_obj)
            score = _pair_score_for_candidate(
                rule=rule,
                distance=float(distance),
                overlap_px=overlap_px,
                iou=iou,
            )
            if score is None:
                continue
            key = (first_obj.cell_id, second_obj.cell_id)
            current = candidates_by_ids.get(key)
            if current is None or score > current.score:
                candidates_by_ids[key] = PairCandidate(first_obj.cell_id, second_obj.cell_id, score)
    return sorted(candidates_by_ids.values(), key=lambda item: item.score, reverse=True)


def _store_pair_score(
    pair_scores: dict[tuple[str, str], dict[tuple[str, str], float]],
    first_channel: str,
    first_id: str,
    second_channel: str,
    second_id: str,
    score: float,
) -> None:
    key = _pair_channel_key(first_channel, second_channel)
    if key[0] == first_channel.upper():
        ids = (first_id, second_id)
    else:
        ids = (second_id, first_id)
    existing = pair_scores[key].get(ids)
    if existing is None or score > existing:
        pair_scores[key][ids] = score


def _lookup_pair_score(
    pair_scores: dict[tuple[str, str], dict[tuple[str, str], float]],
    first_channel: str,
    first_id: str,
    second_channel: str,
    second_id: str,
) -> float | None:
    key = _pair_channel_key(first_channel, second_channel)
    mapping = pair_scores.get(key)
    if not mapping:
        return None
    ids = (first_id, second_id) if key[0] == first_channel.upper() else (second_id, first_id)
    return mapping.get(ids)


def _connected_components(edges: list[tuple[str, str]]) -> list[set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)

    components: list[set[str]] = []
    visited: set[str] = set()
    for node in adjacency:
        if node in visited:
            continue
        stack = [node]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            stack.extend(adjacency[current] - visited)
        components.append(component)
    return components


def _group_candidate_score(
    channels: tuple[str, ...],
    object_ids_by_channel: dict[str, str],
    pair_scores: dict[tuple[str, str], dict[tuple[str, str], float]],
    required_edges: tuple[tuple[str, str], ...],
) -> float:
    pair_sum = 0.0
    for first_channel, second_channel in required_edges:
        score = _lookup_pair_score(
            pair_scores,
            first_channel,
            object_ids_by_channel[first_channel],
            second_channel,
            object_ids_by_channel[second_channel],
        )
        if score is None:
            return -1.0
        pair_sum += score
    return ((len(channels) - 1) * SIZE_BONUS) + pair_sum


def _generate_group_candidates(
    channels: tuple[str, ...],
    objects_by_channel: dict[str, list[DetectedObject]],
    pair_scores: dict[tuple[str, str], dict[tuple[str, str], float]],
    required_edges: tuple[tuple[str, str], ...],
) -> list[GroupCandidate]:
    if len(channels) < 2 or any(not objects_by_channel.get(channel) for channel in channels):
        return []

    canonical_channels = tuple(sorted(channels, key=str.upper))
    search_order = tuple(sorted(canonical_channels, key=lambda channel: len(objects_by_channel[channel])))
    selected: list[DetectedObject] = []
    seen: set[tuple[str, ...]] = set()
    generated: list[GroupCandidate] = []

    def backtrack(index: int) -> None:
        if index >= len(search_order):
            object_ids_by_channel = {channel: obj.cell_id for channel, obj in zip(search_order, selected, strict=False)}
            canonical_ids = tuple(object_ids_by_channel[channel] for channel in canonical_channels)
            if canonical_ids in seen:
                return
            score = _group_candidate_score(canonical_channels, object_ids_by_channel, pair_scores, required_edges)
            if score <= 0:
                return
            seen.add(canonical_ids)
            generated.append(GroupCandidate(canonical_channels, canonical_ids, score))
            return

        current_channel = search_order[index]
        for obj in objects_by_channel[current_channel]:
            compatible = True
            for previous_channel, previous_obj in zip(search_order[:index], selected, strict=False):
                if tuple(sorted((current_channel, previous_channel), key=str.upper)) not in required_edges:
                    continue
                if _lookup_pair_score(
                    pair_scores,
                    current_channel,
                    obj.cell_id,
                    previous_channel,
                    previous_obj.cell_id,
                ) is None:
                    compatible = False
                    break
            if not compatible:
                continue
            selected.append(obj)
            backtrack(index + 1)
            selected.pop()

    backtrack(0)
    return generated


def _select_best_candidate_subset(candidates: list[GroupCandidate]) -> list[GroupCandidate]:
    if not candidates:
        return []

    ordered = sorted(
        candidates,
        key=lambda item: (len(item.channels), item.score, item.object_ids),
        reverse=True,
    )
    suffix_scores = [0.0] * (len(ordered) + 1)
    for index in range(len(ordered) - 1, -1, -1):
        suffix_scores[index] = suffix_scores[index + 1] + max(0.0, ordered[index].score)

    best_score = -1.0
    best_indices: list[int] = []

    def backtrack(index: int, used_ids: set[str], total_score: float, chosen: list[int]) -> None:
        nonlocal best_score, best_indices
        if total_score + suffix_scores[index] <= best_score:
            return
        if index >= len(ordered):
            if total_score > best_score:
                best_score = total_score
                best_indices = list(chosen)
            return

        candidate = ordered[index]
        if not any(object_id in used_ids for object_id in candidate.object_ids):
            used_ids.update(candidate.object_ids)
            chosen.append(index)
            backtrack(index + 1, used_ids, total_score + candidate.score, chosen)
            chosen.pop()
            for object_id in candidate.object_ids:
                used_ids.remove(object_id)
        backtrack(index + 1, used_ids, total_score, chosen)

    backtrack(0, set(), 0.0, [])
    return [ordered[index] for index in best_indices]


def _group_label(channels: tuple[str, ...]) -> str:
    prefixes = {1: "single", 2: "double", 3: "triple", 4: "quadruple"}
    return f"{prefixes.get(len(channels), f'{len(channels)}ch')}_" + "_".join(channels)


def _majority_vote(values: list[Any], default: Any) -> Any:
    if not values:
        return default
    counts: dict[Any, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    return max(counts, key=lambda item: counts[item])


def _build_overlap_group_record(
    *,
    group_id: str,
    objects: list[DetectedObject],
    all_channels: list[str],
) -> OverlapGroupRecord:
    channels = tuple(sorted({_channel_key_for_object(obj) for obj in objects}, key=str.upper))
    channel_flags = {channel: 1 if channel in channels else 0 for channel in all_channels}
    region_ids = [obj.region_id for obj in objects if obj.region_id > 0]
    region_id = int(_majority_vote(region_ids, 0))
    representative = next((obj for obj in objects if obj.region_id == region_id), objects[0])
    hemispheres = [obj.hemisphere for obj in objects]
    hemisphere = str(_majority_vote(hemispheres, "unknown"))
    mean_intensity = float(np.mean([obj.mean_intensity for obj in objects])) if objects else 0.0
    mean_cell_area = float(np.mean([obj.area_px for obj in objects])) if objects else 0.0
    label = _group_label(channels)
    for obj in objects:
        obj.overlap_group = label
        obj.matched_channels = list(channels)
        obj.overlap_group_id = group_id
    return OverlapGroupRecord(
        group_id=group_id,
        animal_id=objects[0].animal_id,
        section_id=objects[0].section_id,
        channels=channels,
        channel_flags=channel_flags,
        objects=objects,
        region_id=region_id,
        region_name=representative.region_name if region_id else "Unassigned",
        hemisphere=hemisphere,
        mean_intensity=mean_intensity,
        mean_cell_area=mean_cell_area,
    )


def _relevant_pairs(available_channels: list[str], specs: list[OverlapSpec], config: MatchingConfig) -> list[tuple[str, str]]:
    if not specs:
        configured_pairs = {
            tuple(sorted(re.findall(r"CH\d+", str(key).upper()), key=str.upper))
            for key in config.pair_rules
            if len(re.findall(r"CH\d+", str(key).upper())) == 2
        }
        return sorted(
            [
                (first_channel, second_channel)
                for first_channel, second_channel in configured_pairs
                if first_channel in available_channels and second_channel in available_channels
            ],
            key=lambda item: (item[0].upper(), item[1].upper()),
        )
    allowed: set[tuple[str, str]] = set()
    available = {channel.upper(): channel for channel in available_channels}
    for channel_set in { _normalize_combo(spec.all_channels) for spec in specs if len(spec.all_channels) >= 2 }:
        if not set(channel_set).issubset(set(available)):
            continue
        display_channels = [available[channel] for channel in channel_set]
        for first_channel, second_channel in itertools.combinations(display_channels, 2):
            if _combo_key((first_channel, second_channel)) in {
                _combo_key(re.findall(r"CH\d+", str(key).upper())) for key in config.pair_rules
            }:
                allowed.add(tuple(sorted((first_channel, second_channel), key=str.upper)))
    return sorted(allowed, key=lambda item: (item[0].upper(), item[1].upper()))


def _assign_default_single_groups(
    section_results: list[SectionChannelResult],
    all_channels: list[str],
) -> list[OverlapGroupRecord]:
    groups: list[OverlapGroupRecord] = []
    counter = 1
    for result in section_results:
        for obj in result.detected_objects:
            group_id = f"group_{counter:06d}"
            counter += 1
            groups.append(
                _build_overlap_group_record(
                    group_id=group_id,
                    objects=[obj],
                    all_channels=all_channels,
                )
            )
    for result in section_results:
        result.overlap_groups = groups
    return groups


def apply_multichannel_matching(
    section_results: list[SectionChannelResult],
    config: MatchingConfig,
) -> pd.DataFrame:
    """Assign overlap groups in-place and return section-level multichannel summary."""

    objects_by_id: dict[str, DetectedObject] = {}
    objects_by_channel: dict[str, list[DetectedObject]] = defaultdict(list)
    for result in section_results:
        key = _channel_key_for_result(result)
        for obj in result.detected_objects:
            objects_by_channel[key].append(obj)
            objects_by_id[obj.cell_id] = obj

    all_channels = sorted(objects_by_channel, key=str.upper)
    empty = pd.DataFrame(
        columns=[
            "animal_id",
            "section_id",
            "combination",
            "region_id",
            "region_name",
            "hemisphere",
            "n_groups",
            "member_channels",
            "n_member_cells",
        ]
    )
    if not objects_by_id:
        return empty

    for obj in objects_by_id.values():
        key = _channel_key_for_object(obj)
        obj.overlap_group = f"single_{key}"
        obj.matched_channels = [key]
        obj.overlap_group_id = ""

    specs = _expand_specs(config) if config.enabled else []
    if not config.enabled or len(objects_by_channel) < 2:
        _assign_default_single_groups(section_results, all_channels)
        return empty
    _validate_required_pair_rules(config, specs)

    pair_scores: dict[tuple[str, str], dict[tuple[str, str], float]] = defaultdict(dict)
    edges: list[tuple[str, str]] = []
    for first_channel, second_channel in _relevant_pairs(all_channels, specs, config):
        rule = _pair_rule(config, first_channel, second_channel)
        candidates = _pair_candidates(objects_by_channel[first_channel], objects_by_channel[second_channel], rule)
        for candidate in candidates:
            _store_pair_score(
                pair_scores,
                first_channel,
                candidate.first_id,
                second_channel,
                candidate.second_id,
                candidate.score,
            )
            edges.append((candidate.first_id, candidate.second_id))

    if not edges:
        _assign_default_single_groups(section_results, all_channels)
        return empty

    relevant_channel_sets = _positive_channel_sets(specs)
    if not relevant_channel_sets:
        relevant_channel_sets = [tuple(pair) for pair in _relevant_pairs(all_channels, specs, config)]

    selected_groups: list[OverlapGroupRecord] = []
    used_object_ids: set[str] = set()
    group_counter = 1

    for component_ids in _connected_components(edges):
        component_objects = [objects_by_id[object_id] for object_id in component_ids]
        component_by_channel: dict[str, list[DetectedObject]] = defaultdict(list)
        for obj in component_objects:
            component_by_channel[_channel_key_for_object(obj)].append(obj)

        candidates: list[GroupCandidate] = []
        for channel_set in relevant_channel_sets:
            if not set(channel_set).issubset(set(component_by_channel)):
                continue
            required_edges = tuple(sorted(_available_rule_edges(config, channel_set), key=lambda item: (item[0].upper(), item[1].upper())))
            if not required_edges:
                continue
            candidates.extend(_generate_group_candidates(channel_set, component_by_channel, pair_scores, required_edges))

        for candidate in _select_best_candidate_subset(candidates):
            objects = [objects_by_id[object_id] for object_id in candidate.object_ids]
            group_id = f"group_{group_counter:06d}"
            group_counter += 1
            selected_groups.append(
                _build_overlap_group_record(
                    group_id=group_id,
                    objects=objects,
                    all_channels=all_channels,
                )
            )
            used_object_ids.update(candidate.object_ids)

    for object_id, obj in objects_by_id.items():
        if object_id in used_object_ids:
            continue
        group_id = f"group_{group_counter:06d}"
        group_counter += 1
        selected_groups.append(
            _build_overlap_group_record(
                group_id=group_id,
                objects=[obj],
                all_channels=all_channels,
            )
        )

    selected_groups.sort(key=lambda group: group.group_id)
    for result in section_results:
        result.overlap_groups = selected_groups

    summary_rows: list[dict[str, object]] = []
    for group in selected_groups:
        if len(group.channels) < 2:
            continue
        summary_rows.append(
            {
                "animal_id": group.animal_id,
                "section_id": group.section_id,
                "combination": _group_label(group.channels),
                "region_id": group.region_id,
                "region_name": group.region_name,
                "hemisphere": group.hemisphere,
                "n_groups": 1,
                "member_channels": ",".join(group.channels),
                "n_member_cells": len(group.objects),
            }
        )

    if not summary_rows:
        return empty
    summary = pd.DataFrame(summary_rows)
    return (
        summary.groupby(
            [
                "animal_id",
                "section_id",
                "combination",
                "region_id",
                "region_name",
                "hemisphere",
                "member_channels",
                "n_member_cells",
            ],
            as_index=False,
        )["n_groups"]
        .sum()
        .sort_values(["animal_id", "section_id", "combination", "region_id"])
    )
