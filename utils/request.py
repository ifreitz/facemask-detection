import json, decimal

def round_off_req_data(s):
    """Return round off request data"""
    return decimal.Decimal(str(round(float(s), 2)))

def extract_json_data(request) -> dict:
    """Extract request data"""
    return json.loads(request.body.decode("utf-8"), parse_float=round_off_req_data) \
        if request.body.decode("utf-8") else {}