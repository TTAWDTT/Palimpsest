"""Memory-bounded selection of descriptor columns from a complete signed cache."""

import csv
import json

from palimpsest.io.hashing import file_sha256
from .pixel_features import audit_variants


def load_signed_columns(directory,names,inventory,*,repository_root,inventory_sha,
                        ordinary_variants,selection_variant):
    receipt_path = directory/'features.json'
    receipt = json.loads(receipt_path.read_text())
    if (not receipt.get('passed',True) or receipt['csv_sha256']!=file_sha256(directory/'features.csv')
            or receipt['inventory_sha256']!=inventory_sha
            or any(file_sha256(repository_root/k)!=v for k,v in receipt['code_pins'].items())):
        raise ValueError('Signed component cache changed')
    wanted = set(inventory[0])|{'variant'}|set(names)
    with (directory/'features.csv').open(encoding='utf-8',newline='') as stream:
        reader = csv.DictReader(stream)
        if not wanted.issubset(reader.fieldnames):
            raise ValueError('Signed descriptor columns missing')
        rows = [{k:r[k] for k in wanted} for r in reader]
    audit_variants(rows,inventory,names,ordinary_variants=ordinary_variants,selection_variant=selection_variant)
    return rows,file_sha256(receipt_path)
