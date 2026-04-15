import re
from typing import Dict, Iterable, List, Sequence


OTP_PATTERNS = [
    re.compile(
        r"\b(?:код|код подтверждения|код верификации|смс[- ]?код|пароль|otp|one[- ]time password)\s*[:\-]?\s*(\d{3,8})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:\b(?:никому не сообщайте|не сообщайте|не передавайте|не называйте)\b).*?(\d{3,8})",
        re.IGNORECASE,
    ),
]

PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?\d[\d\-\s()]{8,}\d)")
CARD_PATTERN = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
SPACE_PATTERN = re.compile(r"\s+")
SPEAKER_PATTERN = re.compile(
    r"^\s*(?:мошенник|звонящий|оператор|ответчик|клиент|собеседник|абонент|user|agent)\s*[:\-]\s*",
    re.IGNORECASE,
)

FEATURE_COLUMNS = [
    "scenario_type",
    "channel",
    "speaker_role_pattern",
    "fraud_stage",
    "has_code_request",
    "has_money_transfer_request",
    "has_urgency",
    "has_threat",
    "has_authority_impersonation",
    "has_sensitive_data_request",
    "has_remote_access_request",
    "victim_confused_or_resisting",
    "risk_markers_count",
]

SCENARIO_PATTERNS = [
    ("safe_account", r"безопасн\w* счет|резервн\w* счет|страхов\w* счет|спецсчет"),
    ("bank_impersonation", r"банк|служб\w* безопасности|техподдержк\w* банка|нацбанк|каспи|сбер|финмонитор"),
    ("sim_block", r"sim|сим|номер отключат|продлить сим|замен\w* финансов\w* номер"),
    ("remote_access", r"anydesk|rustdesk|teamviewer|удаленн\w* доступ|демонстрац\w* экран"),
    ("relative_in_trouble", r"мама|папа|бабуш|дедуш|сын|дочк|внук|родственник|авари|дтп|больниц"),
    ("delivery_fee", r"доставк|курьер|посылк|таможенн|пошлин"),
    ("government_or_police", r"полици|следоват|госуслуг|пенсионн|налог|суд|арест|взыскан"),
    ("medical_or_service", r"клиник|поликлиник|анализ|врач|регистратур"),
]

AUTHORITY_PATTERN = re.compile(r"банк|служб\w* безопасности|оператор|полици|следоват|госуслуг|поддержк|провайдер|пенсионн", re.IGNORECASE)
CODE_REQUEST_PATTERN = re.compile(r"скажите код|назовите код|продиктуйте код|код из смс|смс код|код подтверждения|сообщите код", re.IGNORECASE)
TRANSFER_PATTERN = re.compile(r"оплат|перевед|перевод|доплат|пошлин|штраф|внесите|спецсчет|безопасн\w* счет|резервн\w* счет|страхов\w* счет", re.IGNORECASE)
URGENCY_PATTERN = re.compile(r"срочно|немедленно|прямо сейчас|времени мало|не кладите трубку|не откладывайте|сегодня", re.IGNORECASE)
THREAT_PATTERN = re.compile(r"заблок|арест|взыскан|спишут|кредит|в опасности|подозрительн\w* операц|отключен", re.IGNORECASE)
SENSITIVE_PATTERN = re.compile(r"cvv|кодовое слово|паспорт|иин|инн|номер карты|реквизит|личн\w* данн", re.IGNORECASE)
REMOTE_PATTERN = re.compile(r"anydesk|rustdesk|teamviewer|удаленн\w* доступ|демонстрац\w* экран|установите программ", re.IGNORECASE)
RESISTANCE_PATTERN = re.compile(r"не понимаю|сам перезвоню|почему|не буду|не хочу|откуда|это точно|сомневаюсь", re.IGNORECASE)
FRAUD_STAGE_PATTERNS = [
    ("transfer_request", r"перевед|перевод|оплат|доплат|безопасн\w* счет|резервн\w* счет"),
    ("data_request", r"код из смс|назовите код|скажите код|cvv|номер карты|кодовое слово|реквизит"),
    ("pressure", r"срочно|немедленно|не кладите трубку|времени мало|иначе"),
    ("hook", r"подозрительн\w* операц|оформлен кредит|ваши деньги|ваш номер|ваша карта"),
]


def mask_otps(text: str, token: str = "<CODE>") -> str:
    if not isinstance(text, str):
        return text

    def replace(match: re.Match) -> str:
        return match.group(0).replace(match.group(1), token)

    masked = text
    for pattern in OTP_PATTERNS:
        masked = pattern.sub(replace, masked)
    return masked


def mask_sensitive_data(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = mask_otps(text)
    text = PHONE_PATTERN.sub("<PHONE>", text)
    text = CARD_PATTERN.sub("<CARD>", text)
    text = URL_PATTERN.sub("<URL>", text)
    return text


def split_transcript_lines(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    parts = re.split(r"[\r\n]+", text)
    return [part.strip() for part in parts if part.strip()]


def strip_speaker_prefix(text: str) -> str:
    if not isinstance(text, str):
        return text
    return SPEAKER_PATTERN.sub("", text).strip()


def normalize_text(text: str, drop_speaker_prefix: bool = False) -> str:
    if not isinstance(text, str):
        text = str(text)
    if drop_speaker_prefix:
        text = strip_speaker_prefix(text)
    text = mask_sensitive_data(text)
    text = text.lower().replace("ё", "е")
    text = SPACE_PATTERN.sub(" ", text).strip()
    return text


def prepare_text_for_model(text: str) -> str:
    lines = split_transcript_lines(text)
    if not lines:
        return normalize_text(text)

    normalized_lines = [normalize_text(line) for line in lines]
    return " ".join(normalized_lines)


def infer_channel(text: str) -> str:
    normalized = prepare_text_for_model(text)
    if "\n" in str(text) or re.search(r"(мошенник|ответчик|оператор|клиент|звонящий)\s*[-:]", str(text), re.IGNORECASE):
        return "call"
    if "telegram" in normalized or "whatsapp" in normalized:
        return "messenger"
    return "sms"


def infer_speaker_role_pattern(text: str) -> str:
    raw = str(text)
    if re.search(r"(мошенник|ответчик|оператор|клиент|звонящий)\s*[-:]", raw, re.IGNORECASE):
        return "dialogue"
    return "attacker_only" if infer_channel(text) != "sms" else "single_message"


def infer_scenario_type(text: str, label: str = "") -> str:
    normalized = prepare_text_for_model(text)
    for name, pattern in SCENARIO_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            return name
    if label == "normal":
        return "benign_service"
    return "generic_fraud"


def infer_fraud_stage(text: str, label: str = "") -> str:
    normalized = prepare_text_for_model(text)
    for name, pattern in FRAUD_STAGE_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            return name
    if label == "normal":
        return "benign"
    return "unknown"


def extract_structured_features(text: str, label: str = "") -> Dict[str, object]:
    normalized = prepare_text_for_model(text)
    features: Dict[str, object] = {
        "scenario_type": infer_scenario_type(text, label),
        "channel": infer_channel(text),
        "speaker_role_pattern": infer_speaker_role_pattern(text),
        "fraud_stage": infer_fraud_stage(text, label),
        "has_code_request": int(bool(CODE_REQUEST_PATTERN.search(normalized))),
        "has_money_transfer_request": int(bool(TRANSFER_PATTERN.search(normalized))),
        "has_urgency": int(bool(URGENCY_PATTERN.search(normalized))),
        "has_threat": int(bool(THREAT_PATTERN.search(normalized))),
        "has_authority_impersonation": int(bool(AUTHORITY_PATTERN.search(normalized))),
        "has_sensitive_data_request": int(bool(SENSITIVE_PATTERN.search(normalized))),
        "has_remote_access_request": int(bool(REMOTE_PATTERN.search(normalized))),
        "victim_confused_or_resisting": int(bool(RESISTANCE_PATTERN.search(normalized))),
    }
    features["risk_markers_count"] = int(
        features["has_code_request"]
        + features["has_money_transfer_request"]
        + features["has_urgency"]
        + features["has_threat"]
        + features["has_authority_impersonation"]
        + features["has_sensitive_data_request"]
        + features["has_remote_access_request"]
    )
    return features


def feature_text_prefix(features: Dict[str, object]) -> str:
    ordered = [
        f"scenario={features['scenario_type']}",
        f"channel={features['channel']}",
        f"pattern={features['speaker_role_pattern']}",
        f"stage={features['fraud_stage']}",
        f"code_request={features['has_code_request']}",
        f"money_transfer={features['has_money_transfer_request']}",
        f"urgency={features['has_urgency']}",
        f"threat={features['has_threat']}",
        f"authority={features['has_authority_impersonation']}",
        f"sensitive={features['has_sensitive_data_request']}",
        f"remote_access={features['has_remote_access_request']}",
        f"victim_resists={features['victim_confused_or_resisting']}",
        f"risk_markers={features['risk_markers_count']}",
    ]
    return " ".join(ordered)


def build_model_text(text: str, features: Dict[str, object]) -> str:
    prepared = prepare_text_for_model(text)
    return f"{feature_text_prefix(features)} [text] {prepared}".strip()


def build_turn_windows(lines: Sequence[str], window_size: int = 3, step: int = 1) -> List[str]:
    cleaned = [line.strip() for line in lines if line and line.strip()]
    if not cleaned:
        return []
    if len(cleaned) <= window_size:
        return [" ".join(cleaned)]

    windows = []
    for start in range(0, len(cleaned) - window_size + 1, step):
        windows.append(" ".join(cleaned[start:start + window_size]))
    return windows


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    output = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output
