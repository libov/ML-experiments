import mlflow

def load_latest_model(experiment_name, run_name, mlflow_uri="http://localhost:5000"):

    mlflow.set_tracking_uri(mlflow_uri)

    # 1. find experiment
    exp = mlflow.get_experiment_by_name(experiment_name)

    # 2. search run
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        output_format="list"
    )

    if len(runs) == 0:
        raise ValueError(f"No runs found for experiment '{experiment_name}' with run name '{run_name}'")
    elif len(runs) > 1:
        print(f"Warning: Multiple runs found for experiment '{experiment_name}' with run name '{run_name}'. Using the first one.")

    run = runs[0]

    # 3. get latest logged model (largest step)
    latest_model = max(run.outputs.model_outputs, key=lambda m: m.step)

    model_id = latest_model.model_id
    print("Latest model:", model_id, "at step:", latest_model.step)

    # 4. load model
    model = mlflow.pytorch.load_model(f"models:/{model_id}")
    
    return model
