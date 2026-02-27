import os, json, joblib, numpy as np
#joblib is used to save and load the trained SVM model, numpy is used for array manipulations
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import config

def train_fold(fold:int) -> None:
    output_path = os.path.join(config.features_dir, f"fold_{fold}", "train_features.npz")
    data = np.load(output_path, allow_pickle=True)

    Xtrain, ytrain = data["X_train"], data["y_train"]
    print(f"Fold {fold}: Loaded training features with shape {Xtrain.shape} and labels with shape {ytrain.shape}")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel=config.svm_kernel, probability=True, class_weight="balanced")),
    ])

    param_grid = {"svm__C": config.svm_C_grid}
    if config.svm_kernel == "rbf":
        param_grid["svm__gamma"] = config.svm_gamma_grid
    
    grid_search = GridSearchCV(pipe, param_grid = param_grid, scoring="balanced_accuracy", cv=3, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    os.makedirs(config.models_dir, exist_ok=True)
    model_path = os.path.join(config.models_dir, f"svm_fold_{fold}.joblib")
    joblib.dump(grid_search.best_estimator_, model_path)
    print(f"Fold {fold}: Best SVM model saved to {model_path} with parameters {grid_search.best_params_} and best score {grid_search.best_score_:.3f}")

    meta = {"fold": fold, "best_params": grid_search.best_params_, "best_score_balacc_cv": float(grid_search.best_score_),
            "kernel": config.svm_kernel}
    with open(os.path.join(config.models_dir, f"svm_fold_{fold}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[SVM] Saved model to: {model_path} with Best: {meta}")

def main():
    for fold in range(config.kfolds):
        train_fold(fold)
if __name__ == "__main__":    
    main()