"""Label vocabularies for experiment 007 — the single source of truth.

Two stages, so no model is mapped to a dataset by hand:

    model's native label ──MODELS──► canonical type ──DATASETS──► the dataset's bucket(s), or None

A detection whose canonical type maps to None on a dataset is dropped before
scoring: the dataset does not annotate it, so it can be neither right nor
wrong (exp 006's rule). Gold spans are never dropped, except on Nemotron's
coarse view (see NEMOTRON_GOLD). Fine types map down to coarse buckets,
never the other way. A broad type (the OpenAI filter's `account_number`,
which its card defines to cover every ID number) maps to a tuple of buckets
and matches gold in any of them.

Each model is queried with its own native labels (exp 003: the same model
moves 5–17pp on label strings alone), restricted to the labels whose type the
dataset scores. Where a model has no native label for a bucket, ZERO_SHOT
adds one plain-English label; a dataset's `ask` list adds one for a type the
bucket hides (ai4privacy's DATETIME is mostly times, and a model with only a
date-of-birth label would otherwise never be asked for one).
"""

# ─── Models: native label -> canonical type ─────────────────────

# fastino/gliner2-privacy-filter-PII-multi (the 42 labels on its card).
GLINER2_PII = {
    "person": "PERSON", "full_name": "PERSON",
    "first_name": "FIRST_NAME", "middle_name": "MIDDLE_NAME", "last_name": "LAST_NAME",
    "date_of_birth": "DOB",
    "email": "EMAIL", "phone_number": "PHONE",
    "address": "ADDRESS", "street_address": "STREET_ADDRESS",
    "city": "CITY", "state_or_region": "STATE", "postal_code": "POSTCODE", "country": "COUNTRY",
    "government_id": "GOV_ID", "national_id_number": "NATIONAL_ID",
    "passport_number": "PASSPORT", "drivers_license_number": "DRIVER_LICENSE",
    "license_number": "LICENSE", "tax_id": "TAX_ID", "tax_number": "TAX_ID",
    "bank_account": "ACCOUNT_NUMBER", "account_number": "ACCOUNT_NUMBER",
    "routing_number": "ROUTING_NUMBER", "iban": "IBAN",
    "payment_card": "CARD_NUMBER", "card_number": "CARD_NUMBER",
    "card_expiry": "CARD_EXPIRY", "card_cvv": "CVV",
    "username": "USERNAME", "ip_address": "IP",
    "account_id": "ACCOUNT_ID", "sensitive_account_id": "ACCOUNT_ID",
    "password": "PASSWORD", "secret": "SECRET", "api_key": "SECRET",
    "access_token": "SECRET", "recovery_code": "SECRET",
    "sensitive_date": "DATE", "document_date": "DATE",
    "expiration_date": "DATE", "transaction_date": "DATE",
}

# The 55 Nemotron-PII labels: nvidia/gliner-PII and pii-proxy-ner (exp 005).
NEMOTRON = {
    "account_number": "ACCOUNT_NUMBER", "age": "AGE", "api_key": "SECRET",
    "bank_routing_number": "ROUTING_NUMBER", "biometric_identifier": "BIOMETRIC",
    "blood_type": "MEDICAL", "certificate_license_number": "LICENSE",
    "city": "CITY", "company_name": "ORG", "coordinate": "COORDINATE",
    "country": "COUNTRY", "county": "COUNTY", "credit_debit_card": "CARD_NUMBER",
    "customer_id": "CUSTOMER_ID", "cvv": "CVV",
    "date": "DATE", "date_of_birth": "DOB", "date_time": "DATE_TIME",
    "device_identifier": "DEVICE_ID", "education_level": "DEMOGRAPHIC",
    "email": "EMAIL", "employee_id": "EMPLOYEE_ID", "employment_status": "DEMOGRAPHIC",
    "fax_number": "FAX", "first_name": "FIRST_NAME", "gender": "GENDER",
    "health_plan_beneficiary_number": "INSURANCE_ID", "http_cookie": "SECRET",
    "ipv4": "IP", "ipv6": "IP", "language": "DEMOGRAPHIC", "last_name": "LAST_NAME",
    "license_plate": "VEHICLE_ID", "mac_address": "DEVICE_ID",
    "medical_record_number": "MRN", "national_id": "NATIONAL_ID",
    "occupation": "DEMOGRAPHIC", "password": "PASSWORD", "phone_number": "PHONE",
    "pin": "PIN", "political_view": "DEMOGRAPHIC", "postcode": "POSTCODE",
    "race_ethnicity": "DEMOGRAPHIC", "religious_belief": "DEMOGRAPHIC",
    "sexuality": "DEMOGRAPHIC", "ssn": "SSN", "state": "STATE",
    "street_address": "STREET_ADDRESS", "swift_bic": "SWIFT", "tax_id": "TAX_ID",
    "time": "TIME", "unique_id": "GENERIC_ID", "url": "URL", "user_name": "USERNAME",
    "vehicle_identifier": "VEHICLE_ID",
}

# daslabhq/pii-proxy: the 24-label recipe (modal_train.py default, full Nemotron).
PII_PROXY_24 = {k: NEMOTRON[k] for k in [
    "first_name", "last_name", "medical_record_number", "date_of_birth", "date",
    "date_time", "time", "email", "phone_number", "street_address", "city", "state",
    "county", "country", "ssn", "health_plan_beneficiary_number",
    "certificate_license_number", "biometric_identifier", "url", "unique_id",
    "blood_type", "pin", "password", "swift_bic"]}

# urchade/gliner_multi_pii-v1 (labels listed on its card).
GLINER_MULTI_PII = {
    "person": "PERSON", "organization": "ORG", "phone number": "PHONE",
    "mobile phone number": "PHONE", "landline phone number": "PHONE", "fax number": "FAX",
    "address": "ADDRESS", "postal code": "POSTCODE", "email": "EMAIL", "email address": "EMAIL",
    "passport number": "PASSPORT", "credit card number": "CARD_NUMBER",
    "social security number": "SSN", "tax identification number": "TAX_ID",
    "national id number": "NATIONAL_ID", "identity card number": "ID_CARD",
    "identity document number": "GOV_ID", "driver's license number": "DRIVER_LICENSE",
    "bank account number": "ACCOUNT_NUMBER", "iban": "IBAN", "cvv": "CVV",
    "credit card expiration date": "CARD_EXPIRY", "date of birth": "DOB",
    "health insurance id number": "INSURANCE_ID", "health insurance number": "INSURANCE_ID",
    "insurance number": "INSURANCE_ID", "ip address": "IP", "username": "USERNAME",
    "social media handle": "USERNAME", "registration number": "GENERIC_ID",
    "student id number": "GENERIC_ID", "license plate number": "VEHICLE_ID",
    "vehicle registration number": "VEHICLE_ID", "medication": "MEDICAL",
    "medical condition": "MEDICAL", "blood type": "MEDICAL",
}

# knowledgator/gliner-pii-base-v1.0 (labels listed on its card).
KNOWLEDGATOR_PII = {
    "name": "PERSON", "first name": "FIRST_NAME", "last name": "LAST_NAME",
    "name medical professional": "PERSON", "dob": "DOB", "age": "AGE", "gender": "GENDER",
    "marital status": "DEMOGRAPHIC", "email address": "EMAIL", "phone number": "PHONE",
    "ip address": "IP", "url": "URL", "location address": "ADDRESS",
    "location street": "STREET_ADDRESS", "location city": "CITY", "location state": "STATE",
    "location country": "COUNTRY", "location zip": "POSTCODE",
    "account number": "ACCOUNT_NUMBER", "bank account": "ACCOUNT_NUMBER",
    "routing number": "ROUTING_NUMBER", "credit card": "CARD_NUMBER",
    "credit card expiration": "CARD_EXPIRY", "cvv": "CVV", "ssn": "SSN",
    "passport number": "PASSPORT", "driver license": "DRIVER_LICENSE",
    "username": "USERNAME", "password": "PASSWORD", "vehicle id": "VEHICLE_ID",
    "healthcare number": "INSURANCE_ID", "condition": "MEDICAL", "drug": "MEDICAL",
    "organization medical facility": "ORG",
}

# openai/privacy-filter: 8 categories (BIOES-decoded by its own Viterbi decoder),
# each read the way its model card maps ai4privacy's labels onto them:
# account_number takes every ID number, secret takes passwords, private_url
# takes IP addresses, private_person takes usernames.
OPENAI_PRIVACY_FILTER = {
    "private_person": "PERSON_OR_HANDLE", "private_email": "EMAIL", "private_phone": "PHONE",
    "private_address": "ADDRESS", "private_url": "URL_OR_IP", "private_date": "DATE",
    "account_number": "ACCOUNT_OR_ID", "secret": "SECRET_OR_PASSWORD",
}

# Microsoft Presidio, default recognizers.
PRESIDIO = {
    "PERSON": "PERSON", "EMAIL_ADDRESS": "EMAIL", "PHONE_NUMBER": "PHONE", "URL": "URL",
    "US_SSN": "SSN", "US_ITIN": "TAX_ID", "US_PASSPORT": "PASSPORT",
    "US_DRIVER_LICENSE": "DRIVER_LICENSE", "IP_ADDRESS": "IP", "DATE_TIME": "DATE_TIME",
    "LOCATION": "LOCATION", "CREDIT_CARD": "CARD_NUMBER", "US_BANK_NUMBER": "ACCOUNT_NUMBER",
    "IBAN_CODE": "IBAN", "MEDICAL_LICENSE": "MEDICAL_LICENSE", "UK_NHS": "NATIONAL_ID",
    "NRP": "DEMOGRAPHIC", "CRYPTO": "CRYPTO",
}

# pii-proxy's regex layer (defaultDetectors) and the Thai pack's detector.
PII_PROXY_REGEX = {
    "email": "EMAIL", "phone": "PHONE", "credit_card": "CARD_NUMBER", "ip_address": "IP",
    "url": "URL", "uuid": "UUID", "tracking_number": "TRACKING",
    "passport_number": "PASSPORT", "national_id": "NATIONAL_ID",
    "driver_license": "DRIVER_LICENSE", "id_card": "ID_CARD",
}

MODELS = {
    "gliner2-pii": GLINER2_PII,
    "gliner2-pii-char": GLINER2_PII,
    "nvidia-gliner-pii": NEMOTRON,
    "pii-proxy-ner": NEMOTRON,
    "pii-proxy-24": PII_PROXY_24,
    "gliner-multi-pii": GLINER_MULTI_PII,
    "knowledgator-pii": KNOWLEDGATOR_PII,
    "openai-privacy-filter": OPENAI_PRIVACY_FILTER,
    "openai-privacy-filter-argmax": OPENAI_PRIVACY_FILTER,
    "presidio": PRESIDIO,
    "regex": PII_PROXY_REGEX,
}

# SPY's seven categories as plain label strings — one query set for every
# model, a guess at "the same deterministic mapping for all systems" in
# GLiNER2-PII's report. Used only to reproduce its table.
SPY7 = {"name": "PERSON", "email": "EMAIL", "phone number": "PHONE", "address": "ADDRESS",
        "id number": "GENERIC_ID", "url": "URL", "username": "USERNAME"}

# Plain-English zero-shot labels for buckets a GLiNER model has no native label
# for. Applied only where the model lacks the canonical type natively.
ZERO_SHOT = {
    "url": "URL", "username": "USERNAME", "street address": "STREET_ADDRESS",
    "passport_number": "PASSPORT", "driver license number": "DRIVER_LICENSE",
    "id card number": "ID_CARD", "gender": "GENDER", "title": "TITLE",
    "ip address": "IP", "geographic coordinates": "COORDINATE",
    "password": "PASSWORD", "date": "DATE", "time": "TIME",
}


# ─── Datasets: canonical type -> bucket ─────────────────────────

NAME = {"PERSON": "NAME", "FIRST_NAME": "NAME", "MIDDLE_NAME": "NAME", "LAST_NAME": "NAME"}
ID_NUMBERS = ["SSN", "NATIONAL_ID", "TAX_ID", "GOV_ID", "ID_CARD", "PASSPORT", "DRIVER_LICENSE",
              "LICENSE", "GENERIC_ID", "ACCOUNT_NUMBER", "ACCOUNT_ID", "ROUTING_NUMBER", "IBAN",
              "CARD_NUMBER", "CUSTOMER_ID", "EMPLOYEE_ID", "MRN", "INSURANCE_ID"]

# SPY: 7 coarse labels. ADDRESS is a street address (Faker street_address);
# ID_NUM is any identifying number (SSN, card, IBAN, BBAN, routing, ISBN, MSISDN, ...).
SPY = {
    "gold": {"NAME": "NAME", "EMAIL": "EMAIL", "PHONE_NUM": "PHONE", "URL": "URL",
             "USERNAME": "USERNAME", "ADDRESS": "ADDRESS", "ID_NUM": "ID_NUM"},
    "types": {**NAME, "EMAIL": "EMAIL", "PHONE": "PHONE", "FAX": "PHONE", "URL": "URL",
              "USERNAME": "USERNAME", "ADDRESS": "ADDRESS", "STREET_ADDRESS": "ADDRESS",
              **{t: "ID_NUM" for t in ID_NUMBERS},
              "PERSON_OR_HANDLE": ("NAME", "USERNAME"), "URL_OR_IP": "URL", "ACCOUNT_OR_ID": "ID_NUM"},
    "whole_names": True,  # a full name is one gold span
}

# ai4privacy pii-masking-300k: exp 006's buckets, unchanged.
AI4PRIVACY = {
    "gold": {
        "GIVENNAME1": "NAME", "GIVENNAME2": "NAME", "LASTNAME1": "NAME", "LASTNAME2": "NAME",
        "LASTNAME3": "NAME", "TITLE": "TITLE", "SEX": "SEX", "EMAIL": "EMAIL", "TEL": "PHONE",
        "SOCIALNUMBER": "NATIONAL_ID", "IDCARD": "ID_CARD", "PASSPORT": "PASSPORT",
        "DRIVERLICENSE": "DRIVER_LICENSE", "IP": "IP", "GEOCOORD": "GEOCOORD",
        "USERNAME": "USERNAME", "PASS": "PASSWORD", "TIME": "DATETIME", "DATE": "DATETIME",
        "BOD": "DATETIME", "CITY": "LOCATION", "STATE": "LOCATION", "COUNTRY": "LOCATION",
        "STREET": "LOCATION", "BUILDING": "LOCATION", "POSTCODE": "LOCATION",
        "SECADDRESS": "LOCATION",
    },
    "types": {
        **NAME, "TITLE": "TITLE", "GENDER": "SEX", "EMAIL": "EMAIL", "PHONE": "PHONE",
        "FAX": "PHONE", "SSN": "NATIONAL_ID", "NATIONAL_ID": "NATIONAL_ID",
        "TAX_ID": "NATIONAL_ID", "ID_CARD": "ID_CARD", "GOV_ID": "ID_CARD",
        "GENERIC_ID": "ID_CARD",  # exp 006's charitable stretch (unique_id)
        "PASSPORT": "PASSPORT", "DRIVER_LICENSE": "DRIVER_LICENSE",
        "LICENSE": "DRIVER_LICENSE",  # exp 006's charitable stretch (certificate_license_number)
        "IP": "IP", "COORDINATE": "GEOCOORD", "USERNAME": "USERNAME", "PASSWORD": "PASSWORD",
        "DATE": "DATETIME", "TIME": "DATETIME", "DATE_TIME": "DATETIME", "DOB": "DATETIME",
        **{t: "LOCATION" for t in ["ADDRESS", "STREET_ADDRESS", "CITY", "STATE", "COUNTRY",
                                   "COUNTY", "POSTCODE", "LOCATION"]},
        "PERSON_OR_HANDLE": ("NAME", "USERNAME"), "URL_OR_IP": "IP", "SECRET_OR_PASSWORD": "PASSWORD",
        "ACCOUNT_OR_ID": ("NATIONAL_ID", "ID_CARD", "PASSPORT", "DRIVER_LICENSE"),
    },
    "ask": ["DATE", "TIME"],
}

# The same validation split in its other five languages (build_data.py
# ai4privacy-<lang>): same labels, same buckets.
AI4PRIVACY_LANGUAGES = ["de", "fr", "it", "es", "nl"]

# Nemotron-PII held-out, coarse view: buckets every contender can be asked
# about. Gold spans outside them (age, occupation, religious belief, ...) are
# left out of this view only; the fine view (native models) keeps every span.
NEMOTRON_TYPES = {
    **NAME, "EMAIL": "EMAIL", "PHONE": "PHONE", "FAX": "PHONE", "URL": "URL",
    "USERNAME": "USERNAME", "PASSWORD": "PASSWORD", "IP": "IP",
    "ADDRESS": "ADDRESS", "STREET_ADDRESS": "ADDRESS",
    **{t: "LOCATION" for t in ["CITY", "STATE", "COUNTRY", "COUNTY", "POSTCODE", "LOCATION"]},
    **{t: "DATETIME" for t in ["DATE", "TIME", "DATE_TIME", "DOB"]},
    **{t: "ID_NUM" for t in ID_NUMBERS},
}
NEMOTRON_GOLD = {label: NEMOTRON_TYPES.get(t) for label, t in NEMOTRON.items()}
# The OpenAI filter's broad types on Nemotron (detections only; no gold label maps to them).
NEMOTRON_BROAD = {"PERSON_OR_HANDLE": ("NAME", "USERNAME"), "URL_OR_IP": ("URL", "IP"),
                  "SECRET_OR_PASSWORD": "PASSWORD", "ACCOUNT_OR_ID": "ID_NUM"}

# Thai set (build_thai.ts): names, national IDs, phones, emails, employee codes.
THAI = {
    "gold": {"NAME": "NAME", "NATIONAL_ID": "ID_NUM", "EMPLOYEE_ID": "ID_NUM",
             "PHONE": "PHONE", "EMAIL": "EMAIL"},
    "types": {**NAME, "EMAIL": "EMAIL", "PHONE": "PHONE", **{t: "ID_NUM" for t in ID_NUMBERS},
              "PERSON_OR_HANDLE": "NAME", "ACCOUNT_OR_ID": "ID_NUM"},
    "whole_names": True,
}

DATASETS = {
    "spy": SPY,
    "ai4privacy": AI4PRIVACY,
    **{f"ai4privacy-{lang}": AI4PRIVACY for lang in AI4PRIVACY_LANGUAGES},
    "nemotron": {"gold": NEMOTRON_GOLD, "types": {**NEMOTRON_TYPES, **NEMOTRON_BROAD},
                 "ask": ["DATE", "TIME"]},
    "thai": THAI,
}


def canonical_type(model: str, label: str):
    return MODELS[model].get(label) or ZERO_SHOT.get(label) or SPY7.get(label)


def buckets(model: str, dataset: str, native_label: str) -> tuple:
    """The dataset buckets a detection with this label can match; () = dropped."""
    canonical = canonical_type(model, native_label)
    return as_tuple(DATASETS[dataset]["types"].get(canonical)) if canonical else ()


def as_tuple(b) -> tuple:
    return () if b is None else b if isinstance(b, tuple) else (b,)


def query_labels(model: str, dataset: str, vocab: str = "dataset") -> list:
    """The labels a GLiNER-style model is asked for.

    dataset — native labels whose type the dataset scores, plus a zero-shot
              label for each scored bucket the model has no native label for,
              and for each type in the dataset's `ask` list it lacks.
    full    — every native label: the model as a pii-proxy detector would
              query it, whatever the text.
    """
    native = MODELS[model]
    types = DATASETS[dataset]["types"]
    if vocab == "full":
        return list(native)
    if vocab == "spy7":
        return list(SPY7)
    labels = [label for label, t in native.items() if types.get(t)]
    covered = {b for label in labels for b in as_tuple(types[native[label]])}
    have = set(native.values())
    ask = DATASETS[dataset].get("ask", [])
    labels += [label for label, t in ZERO_SHOT.items() if types.get(t) and label not in native
               and (not set(as_tuple(types[t])) <= covered or (t in ask and t not in have))]
    return labels
