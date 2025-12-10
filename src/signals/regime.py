"""Hidden Markov Model regime detector for market state classification.

This module implements a regime detector using Gaussian Hidden Markov Models
to classify market conditions into three distinct regimes: RISK_ON, NEUTRAL,
and RISK_OFF. The detector analyzes market features (VIX, trend, spreads) to
identify the current market regime and provide probability distributions.

The HMM approach allows for:
- Capturing temporal dynamics and regime persistence
- Probabilistic regime assignments with uncertainty quantification
- Learning regime transition probabilities from historical data

Key Design Decisions:
- State-to-regime mapping is determined post-fitting based on feature means
- Mapping is deterministic once fitted (stored as instance attribute)
- Model validates feature dimensions to prevent prediction errors
"""

import logging
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
from hmmlearn.hmm import GaussianHMM
from pydantic import BaseModel, Field

from src.data.models import Regime

logger = logging.getLogger(__name__)


class RegimeDetectorConfig(BaseModel):
    """Configuration for the HMM regime detector.

    This config controls the HMM parameters and training process.

    Attributes:
        n_states: Number of hidden states in the HMM. Default is 3 to match
            the three market regimes (RISK_ON, NEUTRAL, RISK_OFF).
        n_iterations: Maximum number of EM iterations for fitting.
        covariance_type: Type of covariance matrix for Gaussian emissions.
            Options: 'spherical', 'diag', 'full', 'tied'.
        random_state: Random seed for reproducibility.
    """

    n_states: int = Field(default=3, ge=2, le=10)
    n_iterations: int = Field(default=100, ge=10, le=1000)
    covariance_type: Literal["spherical", "diag", "full", "tied"] = Field(
        default="full"
    )
    random_state: int = Field(default=42)


class RegimeDetectorError(Exception):
    """Base exception for regime detector errors."""

    pass


class NotFittedError(RegimeDetectorError):
    """Raised when prediction is attempted before fitting."""

    pass


class FeatureDimensionError(RegimeDetectorError):
    """Raised when feature dimensions do not match training data."""

    pass


class RegimeDetector:
    """HMM-based market regime detector.

    Uses a Gaussian Hidden Markov Model to detect market regimes from
    feature data. The detector identifies three regimes:

    - RISK_ON: Favorable market conditions (low VIX, positive trend, tight spreads)
    - NEUTRAL: Mixed or transitional market conditions
    - RISK_OFF: Adverse market conditions (high VIX, negative trend, wide spreads)

    The state-to-regime mapping is learned automatically after fitting by
    analyzing the mean feature values of each hidden state.

    Example:
        >>> detector = RegimeDetector(n_states=3, random_state=42)
        >>> detector.fit(training_features)
        >>> current_regime = detector.predict_regime(latest_features)
        >>> regime_probs = detector.predict_regime_probabilities(latest_features)

    Attributes:
        n_states: Number of hidden states in the HMM.
        config: Full configuration object.
        is_fitted: Whether the model has been fitted.
        n_features: Number of features the model was trained on.
    """

    def __init__(
        self,
        n_states: int = 3,
        random_state: int = 42,
        config: RegimeDetectorConfig | None = None,
    ) -> None:
        """Initialize the HMM regime detector.

        Args:
            n_states: Number of hidden states (default 3 for three regimes).
            random_state: Random seed for reproducibility.
            config: Optional full configuration. If provided, n_states and
                random_state parameters are ignored.
        """
        if config is not None:
            self.config = config
        else:
            self.config = RegimeDetectorConfig(
                n_states=n_states,
                random_state=random_state,
            )

        self._model: GaussianHMM | None = None
        self._state_to_regime: dict[int, Regime] | None = None
        self._n_features: int | None = None
        self._feature_means: np.ndarray | None = None
        self._feature_stds: np.ndarray | None = None

    @property
    def n_states(self) -> int:
        """Number of hidden states in the HMM."""
        return self.config.n_states

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self._model is not None and self._state_to_regime is not None

    @property
    def n_features(self) -> int | None:
        """Number of features the model was trained on."""
        return self._n_features

    def _validate_fitted(self) -> None:
        """Validate that the model has been fitted.

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        if not self.is_fitted:
            raise NotFittedError(
                "RegimeDetector must be fitted before prediction. "
                "Call fit() with training data first."
            )

    def _validate_features(self, features: np.ndarray) -> np.ndarray:
        """Validate and reshape feature array.

        Args:
            features: Feature array to validate.

        Returns:
            Validated and reshaped feature array (2D).

        Raises:
            ValueError: If features array is empty.
            FeatureDimensionError: If feature dimensions don't match training.
        """
        if features.size == 0:
            raise ValueError("Features array cannot be empty.")

        # Ensure 2D array
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Check feature dimensions match training
        if self._n_features is not None and features.shape[1] != self._n_features:
            raise FeatureDimensionError(
                f"Feature dimension mismatch. Model trained on {self._n_features} "
                f"features, but received {features.shape[1]} features."
            )

        return features

    def fit(self, features: np.ndarray) -> "RegimeDetector":
        """Fit the HMM on historical feature data.

        Trains the Gaussian HMM on the provided features and then maps
        the learned hidden states to market regimes based on feature
        characteristics.

        The mapping uses the following heuristics based on typical feature
        ordering (VIX-like, trend-like, spread-like):
        - RISK_ON: State with lowest mean of first feature (VIX proxy)
        - RISK_OFF: State with highest mean of first feature
        - NEUTRAL: Remaining state(s)

        Args:
            features: Training features array of shape (n_samples, n_features).
                Rows should be temporally ordered (oldest first).
                Expected features: VIX or volatility measure, trend indicator,
                credit spreads or similar risk measure.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If features array is empty or has insufficient samples.
        """
        # Validate input
        if features.size == 0:
            raise ValueError("Features array cannot be empty.")

        if features.ndim == 1:
            features = features.reshape(-1, 1)

        n_samples, n_features = features.shape

        # Require at least 3x n_states samples for meaningful fitting
        min_samples = self.n_states * 3
        if n_samples < min_samples:
            raise ValueError(
                f"Insufficient training samples. Received {n_samples}, "
                f"but need at least {min_samples} for {self.n_states} states."
            )

        self._n_features = n_features

        # Standardize features for numerical stability
        self._feature_means = np.mean(features, axis=0)
        self._feature_stds = np.std(features, axis=0)
        # Prevent division by zero
        self._feature_stds = np.where(
            self._feature_stds < 1e-8, 1.0, self._feature_stds
        )
        features_standardized = (features - self._feature_means) / self._feature_stds

        # Initialize and fit the HMM
        self._model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.config.covariance_type,
            n_iter=self.config.n_iterations,
            random_state=self.config.random_state,
        )

        self._model.fit(features_standardized)

        logger.info(
            f"HMM fitted on {n_samples} samples with {n_features} features. "
            f"Converged: {self._model.monitor_.converged}"
        )

        # Map states to regimes
        self._map_states_to_regimes()

        return self

    def _map_states_to_regimes(self) -> None:
        """Map HMM hidden states to market regimes based on feature means.

        This method analyzes the emission means of each state to determine
        the appropriate regime mapping. The mapping logic assumes features
        are ordered with volatility/risk measures first:

        - State with lowest mean on first feature -> RISK_ON (low VIX)
        - State with highest mean on first feature -> RISK_OFF (high VIX)
        - Remaining states -> NEUTRAL

        For n_states > 3, the middle states are all mapped to NEUTRAL.
        For n_states < 3, some regimes may not be represented.

        This method should only be called after fitting.
        """
        if self._model is None:
            raise NotFittedError("Cannot map states before fitting.")

        # Get standardized means from the HMM
        state_means = self._model.means_

        # Use the first feature (typically VIX or volatility) for sorting
        # Lower VIX -> RISK_ON, Higher VIX -> RISK_OFF
        first_feature_means = state_means[:, 0]

        # Sort states by first feature mean (ascending)
        sorted_state_indices = np.argsort(first_feature_means)

        # Create deterministic mapping
        self._state_to_regime = {}

        if self.n_states >= 3:
            # RISK_ON: lowest VIX-like feature mean
            self._state_to_regime[sorted_state_indices[0]] = Regime.RISK_ON
            # RISK_OFF: highest VIX-like feature mean
            self._state_to_regime[sorted_state_indices[-1]] = Regime.RISK_OFF
            # NEUTRAL: all middle states
            for i in range(1, self.n_states - 1):
                self._state_to_regime[sorted_state_indices[i]] = Regime.NEUTRAL
        elif self.n_states == 2:
            # Two states: RISK_ON and RISK_OFF only
            self._state_to_regime[sorted_state_indices[0]] = Regime.RISK_ON
            self._state_to_regime[sorted_state_indices[1]] = Regime.RISK_OFF
            logger.warning(
                "Only 2 states configured. NEUTRAL regime will not be predicted."
            )
        else:
            # Single state: map to NEUTRAL
            self._state_to_regime[0] = Regime.NEUTRAL
            logger.warning("Only 1 state configured. All predictions will be NEUTRAL.")

        logger.info(f"State-to-regime mapping established: {self._state_to_regime}")

    def _standardize_features(self, features: np.ndarray) -> np.ndarray:
        """Standardize features using training statistics.

        Args:
            features: Raw feature array.

        Returns:
            Standardized feature array.
        """
        if self._feature_means is None or self._feature_stds is None:
            raise NotFittedError("Model must be fitted before standardizing features.")

        return (features - self._feature_means) / self._feature_stds

    def predict_regime(self, features: np.ndarray) -> Regime:
        """Predict the current market regime from feature data.

        Uses the Viterbi algorithm to find the most likely state sequence,
        then returns the regime corresponding to the final state.

        Args:
            features: Feature array of shape (n_samples, n_features) or
                (n_features,) for a single observation. If multiple samples
                are provided, returns the regime for the last sample.

        Returns:
            Predicted market regime (RISK_ON, NEUTRAL, or RISK_OFF).

        Raises:
            NotFittedError: If the model has not been fitted.
            FeatureDimensionError: If feature dimensions don't match training.
            ValueError: If features array is empty.
        """
        self._validate_fitted()
        features = self._validate_features(features)

        # Standardize features
        features_std = self._standardize_features(features)

        # Get most likely state sequence
        state_sequence = self._model.predict(features_std)  # type: ignore[union-attr]

        # Return regime for the last observation
        final_state = state_sequence[-1]
        return self._state_to_regime[final_state]  # type: ignore[index]

    def predict_regime_probabilities(self, features: np.ndarray) -> dict[Regime, float]:
        """Return probability distribution over regimes.

        Computes the posterior probability of being in each regime given
        the observed features. Uses the forward-backward algorithm.

        Args:
            features: Feature array of shape (n_samples, n_features) or
                (n_features,) for a single observation. If multiple samples
                are provided, returns probabilities for the last sample.

        Returns:
            Dictionary mapping each Regime to its probability.
            Probabilities sum to 1.0.

        Raises:
            NotFittedError: If the model has not been fitted.
            FeatureDimensionError: If feature dimensions don't match training.
            ValueError: If features array is empty.
        """
        self._validate_fitted()
        features = self._validate_features(features)

        # Standardize features
        features_std = self._standardize_features(features)

        # Get state probabilities using forward-backward
        state_posteriors = self._model.predict_proba(features_std)  # type: ignore[union-attr]

        # Get probabilities for the last observation
        final_state_probs = state_posteriors[-1]

        # Aggregate state probabilities by regime
        regime_probs: dict[Regime, float] = {
            Regime.RISK_ON: 0.0,
            Regime.NEUTRAL: 0.0,
            Regime.RISK_OFF: 0.0,
        }

        for state, regime in self._state_to_regime.items():  # type: ignore[union-attr]
            regime_probs[regime] += float(final_state_probs[state])

        return regime_probs

    def get_transition_matrix(self) -> np.ndarray:
        """Return the regime transition probability matrix.

        The transition matrix A[i,j] gives the probability of transitioning
        from state i to state j. This represents the persistence and
        switching dynamics between market regimes.

        Returns:
            Transition probability matrix of shape (n_states, n_states).
            Rows sum to 1.0.

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        self._validate_fitted()
        return self._model.transmat_.copy()  # type: ignore[union-attr]

    def get_stationary_distribution(self) -> dict[Regime, float]:
        """Return the long-run stationary distribution of regimes.

        Computes the stationary distribution of the Markov chain, which
        represents the long-run proportion of time spent in each regime
        assuming the model is correct.

        The stationary distribution pi satisfies: pi = pi @ A
        where A is the transition matrix.

        Returns:
            Dictionary mapping each Regime to its stationary probability.
            Probabilities sum to 1.0.

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        self._validate_fitted()

        # Get transition matrix
        transmat = self.get_transition_matrix()

        # Compute stationary distribution as left eigenvector of transmat
        # The stationary distribution is the eigenvector corresponding to
        # eigenvalue 1 of the transpose of the transition matrix
        eigenvalues, eigenvectors = np.linalg.eig(transmat.T)

        # Find eigenvector for eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])

        # Normalize to sum to 1
        stationary = stationary / stationary.sum()

        # Aggregate by regime
        regime_stationary: dict[Regime, float] = {
            Regime.RISK_ON: 0.0,
            Regime.NEUTRAL: 0.0,
            Regime.RISK_OFF: 0.0,
        }

        for state, regime in self._state_to_regime.items():  # type: ignore[union-attr]
            regime_stationary[regime] += float(stationary[state])

        return regime_stationary

    def save(self, path: str) -> None:
        """Save the fitted model to disk.

        Saves the complete model state including the HMM parameters,
        state-to-regime mapping, and feature standardization statistics.

        Args:
            path: File path for the saved model (typically .pkl extension).

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        self._validate_fitted()

        model_state = {
            "config": self.config.model_dump(),
            "model": self._model,
            "state_to_regime": self._state_to_regime,
            "n_features": self._n_features,
            "feature_means": self._feature_means,
            "feature_stds": self._feature_stds,
        }

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "wb") as f:
            pickle.dump(model_state, f)  # noqa: S301

        logger.info(f"Model saved to {save_path}")

    @classmethod
    def load(cls, path: str) -> "RegimeDetector":
        """Load a fitted model from disk.

        Args:
            path: File path to the saved model.

        Returns:
            Loaded RegimeDetector with fitted model.

        Raises:
            FileNotFoundError: If the model file does not exist.
            ValueError: If the file contains invalid model data.
        """
        load_path = Path(path)

        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")

        with open(load_path, "rb") as f:
            model_state = pickle.load(f)  # noqa: S301  # nosec B301

        # Validate loaded state
        required_keys = {
            "config",
            "model",
            "state_to_regime",
            "n_features",
            "feature_means",
            "feature_stds",
        }
        if not required_keys.issubset(model_state.keys()):
            raise ValueError(
                f"Invalid model file. Missing keys: "
                f"{required_keys - set(model_state.keys())}"
            )

        # Reconstruct the detector
        config = RegimeDetectorConfig(**model_state["config"])
        detector = cls(config=config)
        detector._model = model_state["model"]
        detector._state_to_regime = model_state["state_to_regime"]
        detector._n_features = model_state["n_features"]
        detector._feature_means = model_state["feature_means"]
        detector._feature_stds = model_state["feature_stds"]

        logger.info(f"Model loaded from {load_path}")

        return detector

    def get_state_characteristics(self) -> dict[Regime, dict[str, float]]:
        """Get the learned characteristics of each regime state.

        Returns the mean feature values for each regime, useful for
        understanding what the model has learned about each market state.

        Returns:
            Dictionary mapping regimes to their mean feature values
            (in original scale, not standardized).

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        self._validate_fitted()

        if (
            self._model is None
            or self._feature_means is None
            or self._feature_stds is None
        ):
            raise NotFittedError("Model attributes not properly initialized.")

        characteristics: dict[Regime, dict[str, float]] = {}

        for state, regime in self._state_to_regime.items():  # type: ignore[union-attr]
            # Convert standardized means back to original scale
            state_means_std = self._model.means_[state]
            state_means_original = (
                state_means_std * self._feature_stds + self._feature_means
            )

            state_info = {
                f"feature_{i}": float(state_means_original[i])
                for i in range(len(state_means_original))
            }

            if regime in characteristics:
                # Multiple states map to same regime - average them
                for key in state_info:
                    characteristics[regime][key] = (
                        characteristics[regime][key] + state_info[key]
                    ) / 2
            else:
                characteristics[regime] = state_info

        return characteristics
