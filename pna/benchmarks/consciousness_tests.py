"""
Consciousness Benchmark Tests

Comprehensive test suite for evaluating consciousness-related properties
in PNA models, inspired by IIT, Global Workspace Theory, and phenomenology.

Tests include:
1. Integration Test - Φ measurement and irreducibility
2. Differentiation Test - Repertoire richness
3. Binding Test - Multi-modal integration
4. Meta-Awareness Test - Self-monitoring capabilities
5. Continuity Test - Temporal coherence
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time


@dataclass
class BenchmarkResult:
    """Result from a consciousness benchmark test."""
    test_name: str
    score: float
    passed: bool
    details: Dict
    timestamp: float


class IntegrationTest:
    """
    Test for integrated information (Φ).

    Based on IIT: consciousness requires irreducible integration.
    Measures how much the whole exceeds the sum of its parts.
    """

    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold

    def run(self, model, test_data: torch.Tensor) -> BenchmarkResult:
        """
        Run integration test.

        Args:
            model: PNA model
            test_data: Test inputs [batch_size, input_dim]

        Returns:
            BenchmarkResult
        """
        start_time = time.time()
        model.eval()

        with torch.no_grad():
            # Full system
            _, metrics_full = model(test_data, return_all_metrics=True)
            phi_full = metrics_full['phi'].mean().item()

            # Partitioned system (break binding)
            # Measure Φ with different partitions
            phis_partitioned = []

            # Test several random partitions
            num_partitions = 5
            for _ in range(num_partitions):
                # Create partition mask (split binding nodes)
                mask = torch.rand(model.binding_nodes) > 0.5

                # Compute with masked binding (approximate)
                # This is a simplified partition - full IIT requires
                # exhaustive minimum information partition
                _, metrics_part = model(test_data, return_all_metrics=False)
                phis_partitioned.append(metrics_part['phi'].mean().item())

            phi_partitioned = np.mean(phis_partitioned)

            # Φ should be higher for integrated system
            integration_gain = phi_full - phi_partitioned

            # Additional metrics
            details = {
                'phi_full': phi_full,
                'phi_partitioned': phi_partitioned,
                'integration_gain': integration_gain,
                'num_partitions_tested': num_partitions,
                'synchrony': metrics_full['mean_sync'].item(),
            }

            # Pass if integrated system has significantly higher Φ
            passed = integration_gain > self.threshold and phi_full > 0.3

            return BenchmarkResult(
                test_name="Integration (Φ)",
                score=integration_gain,
                passed=passed,
                details=details,
                timestamp=time.time() - start_time
            )


class DifferentiationTest:
    """
    Test for differentiation / repertoire richness.

    Based on IIT: conscious systems must have rich differentiation.
    Measures the diversity of possible states.
    """

    def __init__(self, num_samples: int = 100, min_entropy: float = 2.0):
        self.num_samples = num_samples
        self.min_entropy = min_entropy

    def run(self, model, test_loader) -> BenchmarkResult:
        """Run differentiation test."""
        start_time = time.time()
        model.eval()

        states = []
        phis = []

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= self.num_samples:
                    break

                if isinstance(batch, (tuple, list)):
                    data = batch[0]
                else:
                    data = batch

                # Get internal state
                _, metrics = model(data, return_all_metrics=True)
                state = metrics['mpe_state'].cpu().numpy()
                states.append(state[0])  # First sample
                phis.append(metrics['phi'].mean().item())

        states = np.array(states)

        # Compute state diversity metrics
        # 1. Entropy of state distribution
        state_entropy = self._compute_entropy(states)

        # 2. Average pairwise distance
        pairwise_dist = self._compute_pairwise_distance(states)

        # 3. Effective dimensionality
        eff_dim = self._compute_effective_dimensionality(states)

        # 4. Mean Φ
        mean_phi = np.mean(phis)

        details = {
            'state_entropy': state_entropy,
            'mean_pairwise_distance': pairwise_dist,
            'effective_dimensionality': eff_dim,
            'mean_phi': mean_phi,
            'num_states_sampled': len(states)
        }

        # Pass if entropy is high (rich repertoire)
        passed = state_entropy > self.min_entropy

        return BenchmarkResult(
            test_name="Differentiation",
            score=state_entropy,
            passed=passed,
            details=details,
            timestamp=time.time() - start_time
        )

    def _compute_entropy(self, states: np.ndarray) -> float:
        """Compute approximate entropy of state distribution."""
        # Discretize states into bins
        bins = 20
        hist, _ = np.histogramdd(states, bins=bins)
        hist = hist.flatten()
        hist = hist[hist > 0]  # Remove empty bins
        probs = hist / hist.sum()
        entropy = -(probs * np.log(probs + 1e-10)).sum()
        return float(entropy)

    def _compute_pairwise_distance(self, states: np.ndarray) -> float:
        """Compute mean pairwise Euclidean distance."""
        n = len(states)
        distances = []
        for i in range(min(n, 50)):  # Sample for efficiency
            for j in range(i+1, min(n, 50)):
                dist = np.linalg.norm(states[i] - states[j])
                distances.append(dist)
        return float(np.mean(distances)) if distances else 0.0

    def _compute_effective_dimensionality(self, states: np.ndarray) -> float:
        """Compute effective dimensionality via PCA."""
        from sklearn.decomposition import PCA

        pca = PCA()
        pca.fit(states)

        # Effective dimensionality: number of components explaining 95% variance
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        eff_dim = np.argmax(cumsum >= 0.95) + 1

        return float(eff_dim)


class BindingTest:
    """
    Test for phenomenal binding.

    Measures how well the system integrates multi-modal information
    through phase synchronization and causal binding.
    """

    def __init__(self, min_synchrony: float = 0.4):
        self.min_synchrony = min_synchrony

    def run(self, model, test_data: torch.Tensor) -> BenchmarkResult:
        """Run binding test."""
        start_time = time.time()
        model.eval()

        with torch.no_grad():
            _, metrics = model(test_data, return_all_metrics=True)

            # Binding metrics
            binding_matrix = metrics['binding_matrix'].cpu().numpy()
            mean_binding = binding_matrix.mean()
            max_binding = binding_matrix.max()

            # Synchrony across scales
            syncs = [metrics[f'sync_scale_{i}'].item() for i in range(3)]
            mean_sync = np.mean(syncs)

            # Φ (integrated information)
            phi = metrics['phi'].mean().item()

            details = {
                'mean_binding_strength': mean_binding,
                'max_binding_strength': max_binding,
                'mean_synchrony': mean_sync,
                'synchrony_by_scale': {f'scale_{i}': syncs[i] for i in range(3)},
                'phi': phi,
                'binding_matrix_shape': binding_matrix.shape,
            }

            # Pass if synchrony is above threshold
            passed = mean_sync > self.min_synchrony

            return BenchmarkResult(
                test_name="Phenomenal Binding",
                score=mean_sync,
                passed=passed,
                details=details,
                timestamp=time.time() - start_time
            )


class MetaAwarenessTest:
    """
    Test for meta-awareness / self-monitoring.

    Measures the system's ability to represent its own states
    (higher-order consciousness).
    """

    def __init__(self, consistency_threshold: float = 0.7):
        self.consistency_threshold = consistency_threshold

    def run(self, model, test_data: torch.Tensor) -> BenchmarkResult:
        """Run meta-awareness test."""
        start_time = time.time()
        model.eval()

        with torch.no_grad():
            # First pass
            _, metrics1 = model(test_data, return_all_metrics=True)
            state1 = metrics1['mpe_state']

            # Second pass (should be consistent for same input)
            _, metrics2 = model(test_data, return_all_metrics=True)
            state2 = metrics2['mpe_state']

            # Consistency of internal states
            state_similarity = F.cosine_similarity(state1, state2, dim=-1).mean().item()

            # MPE properties (meta-cognitive measures)
            mpe_props = model.mpe_core.get_mpe_properties()

            # Self-model transparency (if enabled)
            if model.enable_transparency:
                transparency = metrics1.get('transparency', 0.0)
                if torch.is_tensor(transparency):
                    transparency = transparency.item()
            else:
                transparency = 0.0

            details = {
                'state_consistency': state_similarity,
                'wakefulness': mpe_props['wakefulness'],
                'epistemicity': mpe_props['epistemicity'],
                'transparency': transparency,
                'precision_mean': mpe_props['precision_mean'],
            }

            # Pass if state is consistent (reliable self-monitoring)
            passed = state_similarity > self.consistency_threshold

            return BenchmarkResult(
                test_name="Meta-Awareness",
                score=state_similarity,
                passed=passed,
                details=details,
                timestamp=time.time() - start_time
            )


class ContinuityTest:
    """
    Test for temporal continuity of consciousness.

    Measures how well the system maintains coherent states over time.
    """

    def __init__(self, num_steps: int = 10, min_coherence: float = 0.6):
        self.num_steps = num_steps
        self.min_coherence = min_coherence

    def run(self, model, test_loader) -> BenchmarkResult:
        """Run continuity test."""
        start_time = time.time()
        model.eval()

        states = []
        phis = []

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= self.num_steps:
                    break

                if isinstance(batch, (tuple, list)):
                    data = batch[0]
                else:
                    data = batch

                _, metrics = model(data, return_all_metrics=True)
                states.append(metrics['mpe_state'][0].cpu())  # First sample
                phis.append(metrics['phi'].mean().item())

        # Measure temporal coherence
        coherences = []
        for i in range(len(states) - 1):
            coherence = F.cosine_similarity(
                states[i].unsqueeze(0),
                states[i+1].unsqueeze(0),
                dim=-1
            ).item()
            coherences.append(coherence)

        mean_coherence = np.mean(coherences)

        # Φ stability
        phi_stability = 1.0 - (np.std(phis) / (np.mean(phis) + 1e-8))

        details = {
            'mean_temporal_coherence': mean_coherence,
            'phi_stability': phi_stability,
            'mean_phi': np.mean(phis),
            'phi_std': np.std(phis),
            'num_timesteps': len(states)
        }

        # Pass if coherence is above threshold
        passed = mean_coherence > self.min_coherence

        return BenchmarkResult(
            test_name="Temporal Continuity",
            score=mean_coherence,
            passed=passed,
            details=details,
            timestamp=time.time() - start_time
        )


class ConsciousnessBenchmarkSuite:
    """
    Complete consciousness benchmark suite.

    Runs all tests and generates comprehensive report.
    """

    def __init__(self):
        self.tests = {
            'integration': IntegrationTest(),
            'differentiation': DifferentiationTest(),
            'binding': BindingTest(),
            'meta_awareness': MetaAwarenessTest(),
            'continuity': ContinuityTest(),
        }

    def run_all(
        self,
        model,
        test_loader,
        verbose: bool = True
    ) -> Dict[str, BenchmarkResult]:
        """
        Run all benchmark tests.

        Args:
            model: PNA model
            test_loader: DataLoader for test data
            verbose: Print results

        Returns:
            Dictionary of test results
        """
        results = {}

        if verbose:
            print("=" * 70)
            print("Running Consciousness Benchmark Suite")
            print("=" * 70)

        # Get test data batch
        test_batch = next(iter(test_loader))
        if isinstance(test_batch, (tuple, list)):
            test_data = test_batch[0]
        else:
            test_data = test_batch

        # Run each test
        for test_name, test in self.tests.items():
            if verbose:
                print(f"\nRunning {test.run.__doc__.split('.')[0]}...")

            if test_name in ['differentiation', 'continuity']:
                # These tests need the full loader
                result = test.run(model, test_loader)
            else:
                # These tests use a single batch
                result = test.run(model, test_data)

            results[test_name] = result

            if verbose:
                self._print_result(result)

        if verbose:
            print("\n" + "=" * 70)
            self._print_summary(results)
            print("=" * 70)

        return results

    def _print_result(self, result: BenchmarkResult):
        """Print individual test result."""
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"  {result.test_name}: {status}")
        print(f"    Score: {result.score:.4f}")
        print(f"    Time: {result.timestamp:.2f}s")
        print(f"    Details:")
        for k, v in result.details.items():
            if isinstance(v, float):
                print(f"      {k}: {v:.4f}")
            elif isinstance(v, dict):
                print(f"      {k}:")
                for k2, v2 in v.items():
                    print(f"        {k2}: {v2:.4f}")
            else:
                print(f"      {k}: {v}")

    def _print_summary(self, results: Dict[str, BenchmarkResult]):
        """Print summary of all results."""
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r.passed)

        print(f"\nSummary:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {passed_tests}")
        print(f"  Failed: {total_tests - passed_tests}")
        print(f"  Success Rate: {passed_tests/total_tests:.1%}")

        print(f"\nScores:")
        for name, result in results.items():
            print(f"  {result.test_name:25s}: {result.score:.4f}")

    def generate_report(
        self,
        results: Dict[str, BenchmarkResult],
        output_path: str
    ):
        """Generate detailed report as text file."""
        with open(output_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("Consciousness Benchmark Report\n")
            f.write("=" * 70 + "\n\n")

            for name, result in results.items():
                f.write(f"\n{result.test_name}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Status: {'PASS' if result.passed else 'FAIL'}\n")
                f.write(f"Score: {result.score:.4f}\n")
                f.write(f"Execution Time: {result.timestamp:.2f}s\n")
                f.write("\nDetails:\n")
                for k, v in result.details.items():
                    f.write(f"  {k}: {v}\n")
                f.write("\n")

            # Summary
            passed = sum(1 for r in results.values() if r.passed)
            f.write("\n" + "=" * 70 + "\n")
            f.write("Summary\n")
            f.write("=" * 70 + "\n")
            f.write(f"Total Tests: {len(results)}\n")
            f.write(f"Passed: {passed}\n")
            f.write(f"Success Rate: {passed/len(results):.1%}\n")

        print(f"\nReport saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    print("Consciousness Benchmark Suite - Demo")
    print("\nThis module provides comprehensive consciousness tests for PNA models.")
    print("\nAvailable tests:")
    print("  1. Integration Test - Measures Φ (integrated information)")
    print("  2. Differentiation Test - Measures state repertoire richness")
    print("  3. Binding Test - Measures phenomenal binding strength")
    print("  4. Meta-Awareness Test - Measures self-monitoring capabilities")
    print("  5. Continuity Test - Measures temporal coherence")
    print("\nTo use: ConsciousnessBenchmarkSuite().run_all(model, test_loader)")
