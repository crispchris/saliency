"""
Explanation sensitivity measures the extent of explanation change when the input is
slightly perturbed. It has been shown that the models that have high explanation sensitivity
are prone to adversarial attacks.

sensitivity_max metric measures maximum sensitivity of an explanation
using Monte Carlo sampling-based approximation.

Note that max sensitivity is similar to Lipschitz Continuity metric
however it is more robust and easier to estimate.
Since the explanation, for instance an attribution function,
may not always be continuous, can lead to unbounded Lipschitz continuity.
Therefore the latter isn’t always appropriate.
References: https://arxiv.org/pdf/1901.09392.pdf
also from Captum: https://captum.ai/api/metrics.html
"""
## -------------------
## --- Third-Party ---
## -------------------
import sys
sys.path.append('..')
from captum.metrics import sensitivity_max
from captum.metrics._core.sensitivity import default_perturb_func


def sensitivity_test(explanation_function,
                     inputs,
                     perturb_function = default_perturb_func,
                     perturb_radius: float = 0.02,
                     n_perturb_samples: int = 10,
                     norm_ord: str = 'fro',
                     max_examples_per_batch=10,
                     **kwargs):
    score = sensitivity_max(explanation_func=explanation_function,
                            inputs=inputs,
                            perturb_func=perturb_function,
                            perturb_radius=perturb_radius,
                            n_perturb_samples=n_perturb_samples,
                            norm_ord=norm_ord,
                            max_examples_per_batch=max_examples_per_batch,
                            **kwargs)
    return score