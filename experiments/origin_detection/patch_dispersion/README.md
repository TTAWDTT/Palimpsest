# Frozen patch dispersion — prospective thirty-second iteration

Register before new extraction or selection inspection. Single small DINO
alone was fast but insufficient; global means discard feature spread. Test a
fixed compact second-moment descriptor without backbone training or a second
encoder. Prior second-order pooling is established, not claimed as new.

Same signed2100sources/6300authorimages, fit1260 weakraw/Q90, threshold420
weak24views, selection420/fourencodings. RR reserved remains sealed. Reuse
official pinnedDINOv2-S/14 FP32/TF32off, same shortside256 center224 input.
One model forward gives CLS384,mean_patch384 (EXACT oldstage18 descriptors)
and256patch tokens. L2-normalize each patch token,compute population centered
variance/channel inFP64, sqrt to standard deviation384, L2-normalize and
affine store(.5+.5unit). Degenerate zero spread maps to.5 in every channel;
zero/invalid patch norm refused. Dimension1152,extra384perchannelspread.
No learnedprojection, hidden-layer/probe/channel/severity/normalization search.

This is sqrt(diag covariance), NOT diagonal of matrix square-root or full
MPN-COV. No off-diagonal relationships, Wishart/Gaussian assumption, independent
patch sampling claim or physical invariance guarantee. Patch summaries are
permutation invariant for fixed tokens mathematically, but DINO has position
and global context, so this does not imply image permutation invariance.

Before full extraction: hand vectors with known dispersion,positive gain and
token permutation checks,two equal-mean different-spread distributions,
zero-spread convention and invalid tokens. Original prefix repeat on CUDA,
bounded60image pilot196variants against signedstage18vectors,wrongcachedvalue
control. Full20160vectors must retain old768prefix bit-for-bit. Cache audits
code/weight/runtime/inventory/unique metadata and fileSHA/dimensions.

Fixed four source-risk/mean heads: dispersion/source, joint/source, joint/mean,
joint/source_null. Weakfitraw/Q90=7560rows,sourceT.1 vs0,ridge.01,scale.001,
500iterations/gradient1e-5. Null is the same exact-fit-truth-orthogonal sham
assignment asstage30,not a new random-seed scan or a permutation significance
test. Calibrate same24views,score60groups130pairs. Full integer count audit,
negative BA and quote arithmetic follow. Preserve every AUC control interval.

If accuracy/stability warrants,measure the SAME120nativefile/3repeat actual
decode+preprocessing+oneencoder+moment+headcost;verify cache/live score parity.
No cached-evaluation timing passed as deployment speed. No final neural
training, simulation, externalunsealing,or modified previous baseline runs.
Working BA80%/decline2pp and predict10/E2E50ms are provisional,not user-signed
acceptance; independent real data is still needed before success.
