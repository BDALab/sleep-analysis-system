import re
from dataclasses import dataclass

AGGREGATION_NAMES = {
    "Mean",
    "Median",
    "Min",
    "Max",
    "Slope",
    "SD",
    "MAD",
    "Range",
    "IQR",
    "CV",
}


@dataclass(frozen=True)
class ParsedFeature:
    feature: str
    source: str
    measurement: str
    aggregation: str


@dataclass(frozen=True)
class FeatureFamily:
    family_id: str
    label: str
    domain: str
    role: str
    description: str


FEATURE_FAMILIES = {
    "long_awakenings": FeatureFamily(
        "long_awakenings",
        "Long awakenings",
        "sleep",
        "primary",
        "Wake episodes lasting more than five minutes.",
    ),
    "wake_bouts": FeatureFamily(
        "wake_bouts",
        "Wake-bout frequency",
        "sleep",
        "primary_secondary",
        "Frequency or variability of sleep-to-wake transitions.",
    ),
    "waso": FeatureFamily(
        "waso",
        "Wake after sleep onset",
        "sleep",
        "primary_waso_corrected",
        "Wakefulness occurring after sleep onset and before final awakening.",
    ),
    "post_sleep_wakefulness": FeatureFamily(
        "post_sleep_wakefulness",
        "Wake after sleep offset",
        "sleep",
        "secondary",
        "Wakefulness after final sleep offset.",
    ),
    "sleep_onset_latency": FeatureFamily(
        "sleep_onset_latency",
        "Sleep onset latency",
        "sleep",
        "primary",
        "Time needed to fall asleep.",
    ),
    "sleep_efficiency": FeatureFamily(
        "sleep_efficiency",
        "Sleep efficiency",
        "sleep",
        "primary",
        "Proportion of time in bed spent asleep.",
    ),
    "sleep_fragmentation": FeatureFamily(
        "sleep_fragmentation",
        "Sleep fragmentation",
        "sleep",
        "secondary",
        "Fragmentation or discontinuity of sleep.",
    ),
    "sleep_duration": FeatureFamily(
        "sleep_duration",
        "Sleep duration",
        "sleep",
        "secondary",
        "Total sleep time.",
    ),
    "time_in_bed": FeatureFamily(
        "time_in_bed",
        "Time in bed",
        "sleep",
        "secondary",
        "Time between going to bed and getting out of bed.",
    ),
    "activity_variability": FeatureFamily(
        "activity_variability",
        "Activity variability/dispersion",
        "activity",
        "primary_activity_enhanced",
        "Within-recording spread, variability, or dispersion of activity.",
    ),
    "activity_level": FeatureFamily(
        "activity_level",
        "Activity level/intensity",
        "activity",
        "secondary",
        "Central tendency, percentile, or intensity level of activity.",
    ),
    "activity_extrema_timing": FeatureFamily(
        "activity_extrema_timing",
        "Activity extrema/timing",
        "activity",
        "secondary",
        "Activity minima, maxima, or relative timing/position measures.",
    ),
    "activity_shape_complexity": FeatureFamily(
        "activity_shape_complexity",
        "Activity shape/complexity",
        "activity",
        "secondary",
        "Distribution shape, entropy, or signal-energy measures.",
    ),
    "diary_sleep_quality": FeatureFamily(
        "diary_sleep_quality",
        "Subjective sleep/rest quality",
        "diary_lifestyle",
        "secondary_confounding_sensitive",
        "Subjective sleep-quality or rest-quality diary measures.",
    ),
    "caffeine": FeatureFamily(
        "caffeine",
        "Caffeine exposure/timing",
        "diary_lifestyle",
        "secondary_confounding_sensitive",
        "Caffeine count, rate, amount, or timing diary measures.",
    ),
    "alcohol": FeatureFamily(
        "alcohol",
        "Alcohol exposure/timing",
        "diary_lifestyle",
        "secondary_confounding_sensitive",
        "Alcohol count, rate, amount, or timing diary measures.",
    ),
    "sleeping_pills": FeatureFamily(
        "sleeping_pills",
        "Sleeping-pill use",
        "diary_lifestyle",
        "secondary_confounding_sensitive",
        "Sleeping-pill use diary measures.",
    ),
    "day_sleep": FeatureFamily(
        "day_sleep",
        "Day sleep / naps",
        "diary_lifestyle",
        "secondary_confounding_sensitive",
        "Daytime sleep or nap diary measures.",
    ),
    "other_sleep": FeatureFamily(
        "other_sleep",
        "Other sleep measures",
        "sleep",
        "exploratory",
        "Sleep-derived measures outside the pre-specified primary families.",
    ),
    "other_activity": FeatureFamily(
        "other_activity",
        "Other activity measures",
        "activity",
        "exploratory",
        "Activity-derived measures outside the pre-specified activity families.",
    ),
    "other_diary_lifestyle": FeatureFamily(
        "other_diary_lifestyle",
        "Other diary/lifestyle measures",
        "diary_lifestyle",
        "exploratory",
        "Diary/lifestyle measures outside the pre-specified secondary families.",
    ),
    "other": FeatureFamily(
        "other",
        "Other measures",
        "other",
        "exploratory",
        "Measures not matched by the fixed feature-family mapper.",
    ),
}

PRIMARY_SLEEP_STABLE_FAMILY_IDS = frozenset(
    {
        "long_awakenings",
        "sleep_onset_latency",
        "sleep_efficiency",
        "wake_bouts",
        "waso",
    }
)
ACTIVITY_EXTENSION_STABLE_FAMILY_IDS = frozenset({"activity_variability"})
SECONDARY_LIFESTYLE_FAMILY_IDS = frozenset(
    {
        "diary_sleep_quality",
        "alcohol",
        "caffeine",
        "sleeping_pills",
        "day_sleep",
    }
)


def list_feature_families():
    return list(FEATURE_FAMILIES.values())


def parse_feature_name(feature):
    feature_text = str(feature).strip()
    source = ""
    measurement = feature_text
    aggregation = ""

    parts = feature_text.split(".", 2)
    if len(parts) == 3 and parts[0] in AGGREGATION_NAMES:
        aggregation, source, measurement = parts
        return ParsedFeature(feature_text, source, measurement, aggregation)

    if len(parts) >= 2:
        source = parts[0]
        measurement = ".".join(parts[1:])
        measurement, aggregation = _strip_trailing_aggregation(measurement)
        return ParsedFeature(feature_text, source, measurement, aggregation)

    return ParsedFeature(feature_text, source, measurement, aggregation)


def feature_family_for_feature(feature):
    parsed = parse_feature_name(feature)
    return feature_family_for_parts(parsed.source, parsed.measurement)


def feature_family_id_for_feature(feature):
    return feature_family_for_feature(feature).family_id


def feature_family_label_for_feature(feature):
    return feature_family_for_feature(feature).label


def feature_family_for_parts(source, measurement):
    source_lower = str(source).lower()
    measurement_lower = str(measurement).lower()
    combined_lower = f"{source_lower} {measurement_lower}"

    lifestyle_family = _diary_lifestyle_family(combined_lower)
    if lifestyle_family is not None:
        return FEATURE_FAMILIES[lifestyle_family]

    if source_lower.startswith(("actigraphy", "diary")):
        sleep_family = _sleep_family(measurement_lower)
        return FEATURE_FAMILIES[sleep_family]

    if source_lower == "activity":
        activity_family = _activity_family(measurement_lower)
        return FEATURE_FAMILIES[activity_family]

    return FEATURE_FAMILIES["other"]


def feature_family_label(source, measurement):
    return feature_family_for_parts(source, measurement).label


def feature_family_id(source, measurement):
    return feature_family_for_parts(source, measurement).family_id


def feature_family_metadata(feature):
    parsed = parse_feature_name(feature)
    family = feature_family_for_parts(parsed.source, parsed.measurement)
    return {
        "Features": parsed.feature,
        "Feature family ID": family.family_id,
        "Feature family": family.label,
        "Feature family domain": family.domain,
        "Feature family role": family.role,
        "Source": parsed.source or "unknown",
        "Nightly summary": parsed.aggregation,
        "Measurement": parsed.measurement,
        "Normalized source": parsed.source.endswith("_norm"),
    }


def _strip_trailing_aggregation(measurement):
    match = re.match(
        r"^(?P<measurement>.+?)\s+\((?P<aggregation>Mean|Median|Min|Max|Slope|SD|MAD|Range|IQR|CV)\)$",
        measurement,
    )
    if not match:
        return measurement, ""
    return match.group("measurement"), match.group("aggregation")


def _diary_lifestyle_family(text):
    if "caffeine" in text:
        return "caffeine"
    if "alcohol" in text:
        return "alcohol"
    if "sleeping_pill" in text or "sleeping pill" in text:
        return "sleeping_pills"
    if "day_sleep" in text or "day sleep" in text or "nap" in text:
        return "day_sleep"
    if (
            "rest_quality" in text
            or "rest quality" in text
            or "sleep_quality" in text
            or "sleep quality" in text
    ):
        return "diary_sleep_quality"
    return None


def _sleep_family(measurement_lower):
    rules = (
        ("awakening > 5 minutes", "long_awakenings"),
        ("wake bouts", "wake_bouts"),
        ("wake after sleep onset", "waso"),
        ("wake after sleep offset", "post_sleep_wakefulness"),
        ("sleep onset latency", "sleep_onset_latency"),
        ("sleep efficiency", "sleep_efficiency"),
        ("sleep fragmentation", "sleep_fragmentation"),
        ("total sleep time", "sleep_duration"),
        ("time in bed", "time_in_bed"),
    )
    for keyword, family_id in rules:
        if keyword in measurement_lower:
            return family_id
    return "other_sleep"


def _activity_family(measurement_lower):
    if any(
            keyword in measurement_lower
            for keyword in (
                    "standard deviation",
                    "variance",
                    "median absolute deviation",
                    "index of dispersion",
                    "relative interquartile range",
                    "interquartile range",
                    "relative interpercentile range",
                    "interpercentile range",
                    "interdencile range",
                    "relative variation range",
                    "studentized range",
                    "modulation",
                    "range",
            )
    ):
        return "activity_variability"

    if any(
            keyword in measurement_lower
            for keyword in (
                    "skewness",
                    "kurtosis",
                    "entropy",
                    "teager kaiser",
            )
    ):
        return "activity_shape_complexity"

    if any(
            keyword in measurement_lower
            for keyword in (
                    "relative position",
                    "max",
                    "min",
            )
    ):
        return "activity_extrema_timing"

    if re.search(r"\b(1st|5th|10th|20th|80th|90th|95th|99th) percentile\b", measurement_lower):
        return "activity_level"

    if any(
            keyword in measurement_lower
            for keyword in (
                    "mean",
                    "median",
                    "mode",
                    "harmonic mean",
                    "percentile",
            )
    ):
        return "activity_level"

    return "other_activity"
