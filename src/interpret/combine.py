def combine(converted_data):
    if len(converted_data) < 1:
        raise ValueError(f"No area to combine")
    current = converted_data[0]
    for area in converted_data[1:]:
        current_lowest = min(current["markers"], key=lambda m: m["id"])
        area_lowest = min(area["markers"], key=lambda m: m["id"])

        if current_lowest["id"] <= area_lowest["id"]:
            ref = current
            other = area
        else:
            ref = area
            other = current

        current = unify(ref, other)
    return current

def unify(ref, other):
    common_ref, common_other = lowest_common_marker(ref, other)
    common_ref_pos   = common_ref["position"]
    common_other_pos = common_other["position"]

    diff = common_ref_pos - common_other_pos 

    standardized_other = standardize(other, diff)
    ref["markers"].extend(standardized_other["markers"])
    ref["houses"].extend(standardized_other["houses"])
    ref["paths"].extend(standardized_other["paths"])

    #cleaned_ref = clean(ref)

    return ref

def lowest_common_marker(a, b):
    a_markers = a["markers"]
    b_markers = b["markers"]

    b_ids = {m["id"] for m in b_markers}
    common = [m for m in a_markers if m["id"] in b_ids]
    lowest_common = min(common, key=lambda m: m["id"])
    lowest_common_counterpart = next(m for m in b_markers if m["id"] == lowest_common["id"])
    return lowest_common, lowest_common_counterpart

def standardize(data, diff):
    for marker in data["markers"]:
        marker["position"] += diff
    for house in data["houses"]:
        house["points"] = [point + diff for point in house["points"]]
    for path in data["paths"]:
        path["points"] = [point + diff for point in path["points"]]
    return data

