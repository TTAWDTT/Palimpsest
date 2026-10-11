# CuRe probability consistency — prospective iteration51

Register before real fitting. Source logit variance50 improves the worst decline
to3.89pp but fails2pp. Test source posterior JS, which is bounded and may assign
less penalty to logit changes that preserve confident labels. AugMix (ICLR2020,
arXiv1912.02781v2 main1-7 and official imagenet.py training block read) uses JS
with three augmentations and a neural classifier. We use six released matched
views and a conventional linear head;not AugMix reproduction or new JS theory.

Source logistic riskT.1 + ridge.01||w||² + lambda*weighted JS Bernoulli views.
JS=H(weighted source mean posterior)-weighted individual H. Stable log-space
group means,no probability clipping. Probabilities use raw fitted logits at
zero; final class_threshold may shift the decision boundary. This loss cannot
guarantee final calibrated decision consistency. Uniform or confidently wrong
source predictions can have zero JS;classification risk and absolute BA required.

Frozen signed36 full1024 CuRe,weakraw/Q90 fit7560,threshold2520/24groups,
selection5040/4variants,1260/420/420sources unchanged. Domain half/source/view
equal weights,scale-floor.001. No pixels/encoder/simulation training or retrieval.
Five fixed heads: zero,JS1,JS12,JS12_wrong_source,JS12_source_null.12 follows
the official AugMix nominal coefficient,not a transferred optimum or scan.
Wrong JS source IDs from fixed47 helper within same-class processing strata,
classification source IDs genuine. Null uses signed30 mapping. Final threshold
role true labels used even for null,not raw sigmoid calibration certification.

Nonconvex objective: two fixed LBFGS starts(original source head,allzero),
max500,gradient<=1e-5 and descent from each start required; choose lower objective.
Both stationary checks must pass; no global-optimum claim or random restart search.
Zero delegates existing fitter and must match all5040 saved36source margins.
Source classification risk/JS/objective and both optimizer starts saved separately.

Before real fit: analytical .25/.75 JS and derivative;opposite extreme logits
remain finite;constant uniform JSzero with nonzero labelled gradient;two-step
independent finite-difference gradient for genuine/wrong grouping;known class
fit,zero parity,single/batch/save/load and invalid inputs. Synthetic1024 budget
pilot before5heads,code pinned with controls. Complete60metrics/130pairs,integer
audit/wrongBA,all4null raw AUC intervals,emitall/wrongquote follow. Keep current
development exposure/near-content candidate,RRreserved and external data sealed.
Goal remains BA>=80%,drop<=2pp and independent real validation;no new goal object.
