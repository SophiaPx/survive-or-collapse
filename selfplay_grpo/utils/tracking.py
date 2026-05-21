import os
from pathlib import Path
from typing import Union, List

from verl.utils.tracking import Tracking


class ReasonRLTracking(Tracking):
    def __init__(self, project_name, experiment_name, default_backend: Union[str, List[str]] = 'console', config=None, resume='never', run_id=None, tags: List[str] = None):
        if isinstance(default_backend, str):
            default_backend = [default_backend]
        for backend in default_backend:
            if backend == 'tracking':
                import warnings
                warnings.warn("`tracking` logger is deprecated. use `wandb` instead.", DeprecationWarning)
            else:
                assert backend in self.supported_backend, f'{backend} is not supported'

        self.logger = {}

        def persist_run_id(run):
            run_id_file = os.environ.get("AZR_WANDB_RUN_ID_FILE")
            if not run_id_file:
                return
            try:
                path = Path(run_id_file)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"{run.id}\n", encoding="utf-8")
                print(f"[selfplay-grpo] Persisted W&B run_id={run.id} to {path}")
            except Exception as exc:
                print(f"[selfplay-grpo] Failed to persist W&B run id to {run_id_file}: {exc}")

        if 'tracking' in default_backend or 'wandb' in default_backend:
            import wandb
            wandb_settings = wandb.Settings(start_method="thread", init_timeout=300)
            
            def cleanup_wandb():
                try:
                    wandb.finish(exit_code=0, quiet=True)
                except Exception:
                    pass
                try:
                    wandb.teardown()
                except Exception:
                    pass

            def init_wandb(run_kwargs):
                return wandb.init(
                    project=project_name,
                    settings=wandb_settings,
                    name=experiment_name,
                    config=config,
                    **run_kwargs,
                )

            wandb_kwargs = {}
            if resume == 'must':
                wandb_kwargs = {'resume': 'must', 'id': run_id}
            elif resume == 'allow':
                wandb_kwargs = {'resume': 'allow', 'id': run_id}
            if tags is not None:
                wandb_kwargs['tags'] = tags
            run = None
            try:
                run = init_wandb(wandb_kwargs)
            except wandb.errors.UsageError as exc:
                should_retry_allow = (
                    resume == 'must'
                    and run_id is not None
                    and "has not been initialized" in str(exc)
                )
                if not should_retry_allow:
                    raise
                print(
                    f"[selfplay-grpo] W&B resume='must' failed for run_id={run_id}; "
                    "starting a fresh W&B run instead."
                )
                cleanup_wandb()
                fallback_kwargs = {}
                if tags is not None:
                    fallback_kwargs['tags'] = tags
                try:
                    run = init_wandb(fallback_kwargs)
                except wandb.errors.CommError as comm_exc:
                    print(
                        "[selfplay-grpo] W&B initialization timed out after fallback; "
                        "continuing with console logging only."
                    )
                    print(f"[selfplay-grpo] W&B error: {comm_exc}")
                    cleanup_wandb()
            except wandb.errors.CommError as exc:
                print(
                    "[selfplay-grpo] W&B initialization timed out; "
                    "continuing with console logging only."
                )
                print(f"[selfplay-grpo] W&B error: {exc}")
                cleanup_wandb()

            if run is not None:
                self.run_id = run.id
                persist_run_id(run)
                self.logger['wandb'] = wandb

        if 'console' in default_backend:
            from verl.utils.logger.aggregate_logger import LocalLogger
            self.console_logger = LocalLogger(print_to_console=True)
            self.logger['console'] = self.console_logger
