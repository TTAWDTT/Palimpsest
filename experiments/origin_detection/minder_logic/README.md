# MINDER set-operation audit — prospective finite counterexample

Primary arxiv2411.19117v1 PDF p6 prints union=min>epsilon; p12 prints
intersection=max>epsilon. Inspect formulas rather than treating prose as a
stability theorem. This is a source-reading audit,not a new detector campaign.

Pin before execution: X={0,1},epsilon=1,d1(0,1)=2,d2(0,1)=1/2;
positive two-point metrics satisfy all metric axioms. Expected union=True,
intersection=False,min predicate=False,max predicate=True. Also enumerate nine
positive distance pairs from {1/2,1,2}: corrected min=intersection,max=union
must hold throughout; printed equalities must fail at four asymmetric pairs.
Reject a deliberately changed expected union=False on the known witness.

Use Fraction and integer Boolean logic,no model pixels,no empirical AUC.
Save source/code/output hashes. A counterexample refutes those printed set
identities;it does not refute the implemented minimum algorithm or its measured
results. No independent agent proof grade or general propagation guarantee.

Run `python -m experiments.origin_detection.minder_logic.audit_logic`.
