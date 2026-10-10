"""Fixed representations make their consistency term constant in head parameters."""

from fractions import Fraction
import json
from pathlib import Path

from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR

OUTPUT = WORK_DIR / 'robust_statistics/dcpt_review'
PDF_SHA = 'da8a2bc726727f30d96ac1eb8e036551d3d3f959c81524dc9b8e9c47654ce5c0'


def fixed_loss(theta):
    return (theta - 1) ** 2 + 2 * (theta + 1) ** 2


def verify_constant_offsets(base, changed, points):
    offsets = {changed(t) - base(t) for t in points}
    if len(offsets) != 1:
        raise ValueError('Loss is not constant in head parameters')
    return next(iter(offsets))


def code_pins():
    files = list(Path(__file__).parent.glob('*.py')) + [Path(__file__).parent / 'README.md',
        REPO_ROOT / 'tests/evaluation/test_frozen_loss_audit.py']
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(files)}


def main():
    if (OUTPUT / 'loss_audit.json').exists():
        raise FileExistsError('Preserve frozen-loss receipt')
    controls = json.loads((OUTPUT / 'software_controls.json').read_text(encoding='utf-8'))
    if not controls['passed'] or controls['code_pins'] != code_pins():
        raise ValueError('Frozen-loss artificial controls changed')
    if file_sha256(OUTPUT / 'main.pdf') != PDF_SHA:
        raise ValueError('DCPT printed version differs')
    import torch
    import torch.nn.functional as functional

    torch.set_num_threads(1)
    clean = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64)
    degraded = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.float64)
    feature_term = (1 - functional.cosine_similarity(clean, degraded)).mean()
    gradients = []
    for strength in (0., .5):
        head = torch.tensor([[.25, -.5], [.75, .125]], dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([0, 1])
        ce = functional.cross_entropy(clean @ head, labels) + functional.cross_entropy(degraded @ head, labels)
        gradients.append(torch.autograd.grad(ce + strength * feature_term, head)[0])
    if feature_term.requires_grad or not torch.equal(gradients[0], gradients[1]):
        raise ValueError('Fixed-feature gradient implication failed')
    trainable = torch.tensor(.5, dtype=torch.float64, requires_grad=True)
    moving = torch.stack((trainable, torch.ones_like(trainable)))
    moving_term = 1 - functional.cosine_similarity(moving, torch.tensor([0., 1.], dtype=torch.float64), dim=0)
    moving_gradient = torch.autograd.grad(moving_term, trainable)[0]
    if moving_gradient.item() == 0:
        raise ValueError('Unfrozen feature negative control failed')
    head = torch.zeros((2, 2), dtype=torch.float64, requires_grad=True)
    clean_logp = functional.log_softmax(clean @ head, dim=1).detach()
    degraded_logp = functional.log_softmax(degraded @ head, dim=1)
    clean_p, degraded_p = clean_logp.exp(), degraded_logp.exp()
    kl = (clean_p * (clean_logp - degraded_logp)
          + degraded_p * (degraded_logp - clean_logp)).sum(1).mean()
    kl_gradient = torch.autograd.grad(kl, head)[0]
    if kl.item() != 0 or torch.count_nonzero(kl_gradient).item() != 0:
        raise ValueError('Uniform symmetric-KL counterexample failed')
    points = tuple(Fraction(n, 3) for n in range(-6, 7))
    offset = verify_constant_offsets(fixed_loss, lambda t: fixed_loss(t) + Fraction(1, 2), points)
    value = {'passed': True, 'code_pins': code_pins(), 'pdf_sha256': PDF_SHA,
        'exact_constant_offset': str(offset), 'exact_stationary_minimizer': '-1/3',
        'feature_loss': feature_term.item(), 'feature_loss_requires_grad': feature_term.requires_grad,
        'head_ce_gradient_bit_exact_with_feature_loss': True,
        'head_ce_gradient': gradients[0].tolist(), 'unfrozen_feature_gradient': moving_gradient.item(),
        'uniform_detached_symmetric_kl': kl.item(), 'uniform_kl_head_gradient_all_zero': True,
        'torch_version': torch.__version__, 'torch_path': torch.__file__, 'device': 'cpu',
        'scope': 'Artificial gradients under printed equations;not author code or reported-result reproduction'}
    (OUTPUT / 'loss_audit.json').write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: value[k] for k in ('passed', 'feature_loss',
        'head_ce_gradient_bit_exact_with_feature_loss', 'unfrozen_feature_gradient', 'uniform_detached_symmetric_kl')}))


if __name__ == '__main__':
    main()
