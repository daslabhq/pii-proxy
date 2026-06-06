"""Shared type vocabulary for experiment 006.

Every system's native types map into coarse BUCKETS. Scoring requires
bucket equality + span overlap. Detections that map to None are EXCLUDED
from precision (the dataset doesn't annotate them — e.g. URLs — so they
are unverifiable, not wrong). Ground-truth buckets are never excluded:
recall is always measured against everything the dataset annotates.
"""

# ai4privacy/pii-masking-300k ground-truth labels -> bucket
GROUND_TRUTH = {
    "GIVENNAME1": "NAME", "GIVENNAME2": "NAME",
    "LASTNAME1": "NAME", "LASTNAME2": "NAME", "LASTNAME3": "NAME",
    "TITLE": "TITLE",
    "SEX": "SEX",
    "EMAIL": "EMAIL",
    "TEL": "PHONE",
    "SOCIALNUMBER": "NATIONAL_ID",
    "IDCARD": "ID_CARD",
    "PASSPORT": "PASSPORT",
    "DRIVERLICENSE": "DRIVER_LICENSE",
    "IP": "IP",
    "GEOCOORD": "GEOCOORD",
    "USERNAME": "USERNAME",
    "PASS": "PASSWORD",
    "TIME": "DATETIME", "DATE": "DATETIME", "BOD": "DATETIME",
    "CITY": "LOCATION", "STATE": "LOCATION", "COUNTRY": "LOCATION",
    "STREET": "LOCATION", "BUILDING": "LOCATION", "POSTCODE": "LOCATION",
    "SECADDRESS": "LOCATION",
}

# Microsoft Presidio entity types -> bucket
PRESIDIO = {
    "PERSON": "NAME",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "US_SSN": "NATIONAL_ID",
    "US_ITIN": "NATIONAL_ID",
    "US_PASSPORT": "PASSPORT",
    "US_DRIVER_LICENSE": "DRIVER_LICENSE",
    "IP_ADDRESS": "IP",
    "DATE_TIME": "DATETIME",
    "LOCATION": "LOCATION",
    # Unverifiable against this dataset's annotations:
    "URL": None, "CREDIT_CARD": None, "US_BANK_NUMBER": None,
    "NRP": None, "MEDICAL_LICENSE": None, "CRYPTO": None,
    "IBAN_CODE": None, "UK_NHS": None,
}

# pii-proxy regex detector types -> bucket
PII_PROXY_REGEX = {
    "email": "EMAIL",
    "phone": "PHONE",
    "ip_address": "IP",
    "passport_number": "PASSPORT",
    "national_id": "NATIONAL_ID",
    "driver_license": "DRIVER_LICENSE",
    "id_card": "ID_CARD",
    # Unverifiable against this dataset's annotations:
    "credit_card": None, "uuid": None, "url": None, "tracking_number": None,
}

# Our fine-tuned GLiNER (exp 004) native fine labels -> bucket
GLINER_NATIVE = {
    "first_name": "NAME", "last_name": "NAME",
    "email": "EMAIL",
    "phone_number": "PHONE",
    "ssn": "NATIONAL_ID",
    "street_address": "LOCATION", "city": "LOCATION", "state": "LOCATION",
    "country": "LOCATION", "county": "LOCATION",
    "date": "DATETIME", "time": "DATETIME",
    "date_of_birth": "DATETIME", "date_time": "DATETIME",
    "password": "PASSWORD",
    # Approximate matches, see README:
    "unique_id": "ID_CARD",
    "certificate_license_number": "DRIVER_LICENSE",
    # Unverifiable / out of scope for this dataset:
    "url": None, "pin": None, "swift_bic": None,
    "medical_record_number": None, "health_plan_beneficiary_number": None,
    "biometric_identifier": None, "blood_type": None,
}

# pii-proxy-ner (exp 005: gliner_small fine-tuned on FULL Nemotron-PII,
# 55 native labels) -> bucket. Unlisted labels (api_key, cvv, occupation,
# political_view, ...) map to None: unverifiable against this dataset.
GLINER_005 = {
    "first_name": "NAME", "last_name": "NAME",
    "email": "EMAIL",
    "phone_number": "PHONE", "fax_number": "PHONE",
    "ssn": "NATIONAL_ID", "national_id": "NATIONAL_ID", "tax_id": "NATIONAL_ID",
    "street_address": "LOCATION", "city": "LOCATION", "state": "LOCATION",
    "country": "LOCATION", "county": "LOCATION", "postcode": "LOCATION",
    "date": "DATETIME", "time": "DATETIME",
    "date_of_birth": "DATETIME", "date_time": "DATETIME",
    "password": "PASSWORD",
    "user_name": "USERNAME",
    "gender": "SEX",
    "ipv4": "IP", "ipv6": "IP",
    "coordinate": "GEOCOORD",
    # Approximate matches, same stretches as exp-004 (see README):
    "unique_id": "ID_CARD",
    "certificate_license_number": "DRIVER_LICENSE",
}

# Zero-shot extension labels (NOT in exp 004 training vocab) -> bucket.
# Tests whether the fine-tuned bi-encoder generalizes off-vocabulary.
GLINER_EXTENDED = {
    "username": "USERNAME",
    "passport_number": "PASSPORT",
    "driver license number": "DRIVER_LICENSE",
    "id card number": "ID_CARD",
    "gender": "SEX",
    "title": "TITLE",
    "ip address": "IP",
    "geographic coordinates": "GEOCOORD",
}

ALL_BUCKETS = sorted(set(GROUND_TRUTH.values()))
