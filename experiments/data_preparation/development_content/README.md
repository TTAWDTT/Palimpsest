# Development original-content audit

Prospective data audit, no detector fitting. Use the signed16 native inventory
and previous role-structure validator, then only2100 original stored-orientation
images. Verify current encoded SHA against its cached SHA, dimensions, and
hash dimension-framed decoded RGB bytes. Reuse Chimera's existing64-bit grey
DCT pHash implementation. No copied hashing recipe or role reassignment.

Enumerate all distinct original-source pairs, save exact decoded RGB groups
and pHash Hamming<=6 candidates, cross-role and conflicting-label counts.
Threshold6 is a screening rule fixed before pixels, not calibrated recall.
Flat/low-information images can collide; pHash similarity is not a confirmed
duplicate. No candidates would not prove near-duplicate-free independence.
Original-only scope does not certify processed/cropped views or model pretrain.

Before real decoding: same RGB under different PNG byte encodings yields one
exact RGB group; planted cross-role/conflicting-label counters; close flat
hashes with different RGB show candidate≠identity; invalid/duplicate source
identity refused. Wrong expected group counts must be rejected. Record
controls and code pins before audit. Keep output under
work/robust_statistics/development_content; do not open external or RR reserved.
