# Exposed development role and byte-identity audit

Register before reading the signed native metadata. Inspect the existing
iteration16 source inventory only; no image decoding, new pixels, refitting,
role reassignment, or deletion. Parent receipts, code pins and feature CSV
digest must match through the existing parent_data path.

Expected6300 native files,2100 sources,three native conditions each. Source
roles1260fit/420threshold/420selection,source label and scene must agree across
conditions. The same source cannot occupy different roles. All record identity,
role,label,domain,condition,SHA256 and positive dimensions checks are explicit.

Group original-only and all-native records by stored byte SHA256. Report all
multi-record groups, distinct-source groups, cross-role groups and conflicting
label groups, including cross-domain identities. Within-source identical
original/processed bytes are separately retained. Finding duplicates does not
authorize changing historical roles or silently removing records.

Before actual metadata: artificial clean panels and planted exact-byte
duplicates across roles/domains/labels, within-source identical views, duplicate
record/source-role conflicts and invalid digest refusal. Use the actual audit
function, save controls and code pins, then run once. Altering a expected
cross-role duplicate count must be refused by the assertion control.

This audits cached byte hashes only. It cannot detect equivalent decoded
pixels, resized/near duplicates, mislabeled content, or establish independent
scientific validation. A clean result cannot explain null AUC anomalies.

Command: `.venv/Scripts/python.exe -X utf8 -m experiments.data_preparation.development_roles.audit_roles`
