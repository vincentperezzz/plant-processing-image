TIPS = {
    ("tomato", "healthy"): "This tomato plant looks healthy. Keep watering even, leave space between plants, and scan again if new spots show up.",
    ("tomato", "mild"): "This tomato plant looks a bit damaged. Pick off the worst leaves, keep foliage drier, and check the rest of the plant in a few days.",
    ("tomato", "critical"): "This tomato plant looks in serious trouble. Isolate it if you can, strip the worst leaves, and ask a local ag tech before it spreads down the row.",
    ("tomato", "dead"): "This tomato tissue looks beyond recovery. Remove it from the field so it does not feed pests or disease on the healthy plants.",
    ("sili", "healthy"): "This sili plant looks healthy. Keep weeds down and watch for curling or spots after rain.",
    ("sili", "mild"): "This sili plant looks a bit damaged. Remove badly marked leaves, avoid wetting the canopy, and check nearby plants.",
    ("sili", "critical"): "This sili plant looks in serious trouble. Curling or heavy damage like this often spreads. Flag the plant and get a local recommendation fast.",
    ("sili", "dead"): "This sili tissue looks too far gone. Pull it and keep the rest of the bed clean.",
    ("eggplant", "healthy"): "This eggplant plant looks healthy. Keep soil moisture steady and scan again if leaves yellow or droop.",
    ("eggplant", "mild"): "This eggplant plant looks a bit damaged. Remove the worst leaves and check stems and nearby plants for more spots or bugs.",
    ("eggplant", "critical"): "This eggplant plant looks in serious trouble. Wilt or heavy mosaic-type damage can take the whole plant. Do not wait — ask a local ag tech.",
    ("eggplant", "dead"): "This eggplant tissue looks beyond recovery. Remove it so neighboring plants stay cleaner.",
    ("palay", "healthy"): "This palay stand looks healthy. Keep walking the field after rain and scan again if tips brown or yellow.",
    ("palay", "mild"): "This palay looks a bit damaged. Note the patch in the field, keep water even, and check again in a few days to see if it spreads.",
    ("palay", "critical"): "This palay looks in serious trouble. Damage like this can move through a paddy. Mark the area and talk to a local rice technician.",
    ("palay", "dead"): "This palay tissue looks too far gone. Remove badly dead hills if practical and watch the rest of the stand.",
    ("lettuce", "healthy"): "This lettuce looks healthy. Keep it cool and clean, and scan again if it turns slimy or spotted.",
    ("lettuce", "mild"): "This lettuce looks a bit damaged. Harvest around the worst heads if you can, keep leaves drier, and check the bed tomorrow.",
    ("lettuce", "critical"): "This lettuce looks in serious trouble. Do not pack damaged heads with clean ones. Clear the worst plants and ask for local advice.",
    ("lettuce", "dead"): "This lettuce looks beyond recovery. Pull it so rot does not spread through the bed.",
}

RETAKE = "The photo is too unclear to trust. Move closer to one leaf, fill the frame, and scan again."
NOT_LEAF = "This does not look like a plant. Point the camera at the crop so leaves or fruit fill the frame, and scan again."
NOT_IN_LIST = "This is not one of the crops I know (palay, sili, tomato, eggplant, lettuce). If it should be, retake the photo closer to one leaf."
UNSURE = "Not sure what this is. Closest guess is weak. Move closer to one leaf so it fills the frame, then scan again."


def tip_for(crop: str | None, health: str | None, unknown: bool, reason: str | None = None) -> str:
    if reason == "not_a_leaf":
        return NOT_LEAF
    if reason == "not_in_list":
        return NOT_IN_LIST
    if unknown or crop is None:
        return UNSURE if reason in {"low_confidence", "low_margin"} else RETAKE
    return TIPS.get((crop, health or "mild"), RETAKE)
