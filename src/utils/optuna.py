"""
Optuna hyperparameter optimization utilities.

Provides OptunaCallback for integration with Model.train() and OptunaOptimizer
for running hyperparameter optimization studies.
"""

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import optuna
from torch.utils.data import Dataset

if TYPE_CHECKING:
    import wandb
    from models.Model import TrainingCallback


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class OptunaStudyConfig:
    """Configuration for running an Optuna study."""

    n_trials: int = 50
    study_name: str = "model_optimization"
    storage: str | None = None  # e.g., "sqlite:///optuna_study.db"
    pruning: bool = True
    metric: str = "f1"  # Key in epoch_metrics to optimize
    direction: str = "maximize"


# =============================================================================
# Callback
# =============================================================================

class OptunaCallback:
    """
    Training callback that reports metrics to Optuna and handles pruning.

    Implements the TrainingCallback protocol for use with Model.train().
    Optionally integrates with Weights & Biases for logging.
    """

    def __init__(
        self,
        trial: optuna.Trial,
        metric: str = "f1",
        pruning: bool = True,
        wandb_project: str | None = None,
        wandb_group: str | None = None,
        model_config: dict | None = None,
    ):
        """
        Initialize the callback.

        Args:
            trial: Optuna trial object
            metric: Key in epoch_metrics to report to Optuna (e.g., "f1", "val_loss")
            pruning: Whether to check for pruning after each epoch
            wandb_project: W&B project name (enables W&B logging if provided)
            wandb_group: W&B group name for organizing trials (typically study name)
            model_config: Model configuration to log to W&B
        """
        self.trial = trial
        self.metric = metric
        self.pruning = pruning
        self.best_value = 0.0
        self.wandb_run = None

        # Initialize W&B if project provided
        if wandb_project:
            import wandb

            # Combine trial params with model config
            config = dict(trial.params)
            if model_config:
                config.update(model_config)
            config["trial_number"] = trial.number

            self.wandb_run = wandb.init(
                project=wandb_project,
                config=config,
                name=f"trial-{trial.number}",
                group=wandb_group,
                tags=["optuna"],
                reinit=True,
            )

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> bool:
        """
        Report metrics to Optuna and check for pruning.

        Args:
            epoch: Current epoch number (0-indexed)
            metrics: Dict of metrics from this epoch

        Returns:
            True to continue training, False to stop (never returns False,
            raises TrialPruned instead)

        Raises:
            optuna.TrialPruned: If the trial should be pruned
        """
        value = metrics.get(self.metric, 0.0)
        if value > self.best_value:
            self.best_value = value

        # Report to Optuna
        self.trial.report(value, epoch)

        # Log to W&B if enabled
        if self.wandb_run:
            import wandb

            wandb.log(metrics, step=epoch)
            wandb.run.summary["best_value"] = self.best_value

        # Check for pruning
        if self.pruning and self.trial.should_prune():
            raise optuna.TrialPruned()

        return True

    def finish(self):
        """Finish the W&B run if active."""
        if self.wandb_run:
            import wandb

            wandb.finish()


# =============================================================================
# Optimizer
# =============================================================================

class OptunaOptimizer:
    """
    Generic Optuna optimizer that works with any Model subclass with OptunaMixin.

    Example usage:
        search_space = ESMSearchSpace(
            hidden_layer_choices=[[512], [1024]],
            dropout_range=(0.1, 0.3),
        )
        study_config = OptunaStudyConfig(n_trials=50)

        optimizer = OptunaOptimizer(ESMModel, search_space, study_config)
        study = optimizer.run_study(train_dataset, val_dataset)
    """

    def __init__(
        self,
        model_class: type,
        search_space: Any,
        study_config: OptunaStudyConfig,
        wandb_project: str | None = None,
    ):
        """
        Initialize the optimizer.

        Args:
            model_class: Model class that has OptunaMixin (must have from_trial classmethod)
            search_space: Search space object appropriate for the model class
            study_config: Configuration for the Optuna study
            wandb_project: W&B project name (enables W&B logging if provided)
        """
        self.model_class = model_class
        self.search_space = search_space
        self.study_config = study_config
        self.wandb_project = wandb_project

    def create_objective(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        epochs: int = 20,
        seed: int = 42,
        num_workers: int = 4,
    ):
        """
        Create an objective function for Optuna.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of training epochs per trial
            seed: Random seed
            num_workers: Number of data loader workers

        Returns:
            Callable objective function for Optuna
        """
        def objective(trial: optuna.Trial) -> float:
            # Create model from trial
            model = self.model_class.from_trial(
                trial,
                self.search_space,
                epochs=epochs,
                seed=seed,
                num_workers=num_workers,
            )

            # Create callback for pruning (with optional W&B logging)
            callback = OptunaCallback(
                trial=trial,
                metric=self.study_config.metric,
                pruning=self.study_config.pruning,
                wandb_project=self.wandb_project,
                wandb_group=self.study_config.study_name if self.wandb_project else None,
            )

            try:
                # Train with callback
                model.train(train_dataset, val_dataset, callback=callback)
            finally:
                # Always finish W&B run (even if pruned)
                callback.finish()

            return callback.best_value

        return objective

    def run_study(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        epochs: int = 20,
        seed: int = 42,
        num_workers: int = 4,
    ) -> optuna.Study:
        """
        Run the full Optuna optimization study.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of training epochs per trial
            seed: Random seed
            num_workers: Number of data loader workers

        Returns:
            Completed Optuna study with best hyperparameters
        """
        objective = self.create_objective(
            train_dataset,
            val_dataset,
            epochs=epochs,
            seed=seed,
            num_workers=num_workers,
        )

        # Create pruner
        pruner = (
            optuna.pruners.MedianPruner()
            if self.study_config.pruning
            else optuna.pruners.NopPruner()
        )

        # Create or load study
        study = optuna.create_study(
            study_name=self.study_config.study_name,
            storage=self.study_config.storage,
            direction=self.study_config.direction,
            pruner=pruner,
            load_if_exists=True,
        )

        # Run optimization
        study.optimize(
            objective,
            n_trials=self.study_config.n_trials,
            show_progress_bar=True,
        )

        # Print results
        print("\n" + "=" * 60)
        print("Optuna Study Complete")
        print("=" * 60)
        print(f"Best trial {self.study_config.metric}: {study.best_trial.value:.4f}")
        print("\nBest hyperparameters:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")

        return study
