ALLOWED_ADJUSTMENT_COVARIATES = ("age", "gender", "education")


def build_scenario_covariate_mapping(preparation, expected_scenarios):
    scenario_rows = preparation.get("scenarios")
    if not isinstance(scenario_rows, list):
        raise ValueError("Preparation manifest is missing the scenarios list")

    mapping = {}
    for scenario in scenario_rows:
        key = (
            tuple(scenario.get("positive_codes", ())),
            tuple(scenario.get("negative_codes", ())),
        )
        if key in mapping:
            raise ValueError(f"Duplicate prepared classification scenario: {key}")
        mapping[key] = tuple(scenario.get("selected_covariates", ()))

    return validate_scenario_covariate_mapping(mapping, expected_scenarios)


def validate_scenario_covariate_mapping(mapping, expected_scenarios):
    if mapping is None:
        raise ValueError(
            "Scenario-specific covariates are required. "
            "Run the classifier through the prepared classification entry point."
        )

    expected_keys = {
        (tuple(positive_codes), tuple(negative_codes))
        for positive_codes, negative_codes in expected_scenarios
    }
    provided_keys = set(mapping)
    missing = expected_keys - provided_keys
    unexpected = provided_keys - expected_keys
    if missing or unexpected:
        raise ValueError(
            "Invalid scenario-specific covariate mapping. "
            f"Missing scenarios: {sorted(missing)}; "
            f"unexpected scenarios: {sorted(unexpected)}"
        )

    normalized = {}
    for key in expected_keys:
        selected = tuple(dict.fromkeys(mapping[key]))
        unknown = set(selected) - set(ALLOWED_ADJUSTMENT_COVARIATES)
        if unknown:
            raise ValueError(
                f"Scenario {key} contains unsupported adjustment covariates: "
                f"{sorted(unknown)}"
            )
        normalized[key] = selected
    return normalized


def resolve_adjustment_columns(
        selected_covariates,
        available_columns,
        covariate_columns,
):
    selected = tuple(dict.fromkeys(selected_covariates))
    unknown = set(selected) - set(ALLOWED_ADJUSTMENT_COVARIATES)
    if unknown:
        raise ValueError(
            f"Unsupported classification adjustment covariates: {sorted(unknown)}"
        )

    missing_mappings = [
        covariate for covariate in selected if covariate not in covariate_columns
    ]
    if missing_mappings:
        raise ValueError(
            "Missing column mappings for adjustment covariates: "
            f"{missing_mappings}"
        )

    available = set(available_columns)
    resolved = [covariate_columns[covariate] for covariate in selected]
    missing_columns = [column for column in resolved if column not in available]
    if missing_columns:
        raise ValueError(
            "Prepared classification data is missing selected adjustment columns: "
            f"{missing_columns}"
        )
    return resolved
