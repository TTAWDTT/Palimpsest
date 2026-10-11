# Slow subspace — prospective twentieth iteration

Mechanism reading: Wiskott and Sejnowski2002, Neural Computation14:715–770,
Sections2–3 equations2.1–2.4,3.11–3.22 and discussion5.1/5.2 read from author PDF.
https://papers.cnl.salk.edu/PDFs/Slow%20Feature%20Analysis_%20Unsupervised%20Learning%20of%20Invariances%202002-3430.pdf
Finite same-source processing differences replace time derivatives; this is a
linear empirical adaptation,not temporal SFA reproduction or a new principle.

Retain CLIP768 signed cache from stage17. Fit1260sources with raw/Q90 in three
real conditions,7560rows,equal domain/source/views. Compare three fixed64D maps:
PCA-largest variance,smallest whitened paired covariance,or one protected
whitened fit class-mean direction plus63slow orthogonal directions. Covariance
eigenvalues below1e-6oflargest discarded;refuse rank<64. Dimension/floor fixed
before metrics;no dimension sweep. All15distinct view differences per fit source,
including cross-condition/encoding;fit-only covariance,never Q70/Q60 or selection.

Each map gets mean or source-temperature.1 logistic head,ridge.01. Fit-only
class-mean energy and paired variation reported;variance alone is not class
information. Null uses source-shuffled labels both in protected map and readout.
Same24raw/Q90 threshold-role calibration and420source selection four encodings.
Keep17weak/source external-to-this-round reference,not independent test.

Known diagonal controls must show stable classless direction,protected yet
varying class direction and healthy invariant class. Refuse wrong source labels.
Collapsing map+linear head into native768 linear rule must agree within1e-10
and exactly preserve all selection decisions;single/batch native score exact.
Fit-only costpilot before full comparison. No new pixels/backbone training,
RRreserved remains sealed. Maximum scene drop,absolute BA,CI and flips required;
provisional80%/2pp cannot be certified with exposed development alone.

Method selection fixed: greatest minimum domain/scene BA,then smaller worst
scene drop,then PCA,then mean head. A failed round does not stop the goal.
