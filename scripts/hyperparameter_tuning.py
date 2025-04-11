import optuna
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import pandas as pd
from datasets import Dataset
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score
import os
import shutil
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
from classification import prepare_dataset, compute_metrics

def objective(trial):
    """Optuna objective function for hyperparameter tuning"""
    # Load data
    train_df = pd.read_csv("processed_train.csv")
    valid_df = pd.read_csv("processed_valid.csv")
    
    # Define hyperparameter search space
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.05)
    batch_size = trial.suggest_categorical("batch_size", [16, 24, 32])
    num_epochs = trial.suggest_int("num_epochs", 2, 3)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.05, 0.15)
    label_smoothing = trial.suggest_float("label_smoothing", 0.1, 0.2)
    hidden_dropout = trial.suggest_float("hidden_dropout", 0.1, 0.3)
    attention_dropout = trial.suggest_float("attention_dropout", 0.1, 0.3)
    
    # Initialize model and tokenizer
    model_name = "cardiffnlp/twitter-roberta-large-2022-154m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add special tokens to tokenizer
    special_tokens = {'additional_special_tokens': ['[CONTEXT]', '[POST]']}
    num_added_tokens = tokenizer.add_special_tokens(special_tokens)
    
    # Initialize model with trial hyperparameters
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        ignore_mismatched_sizes=True,
        hidden_dropout_prob=hidden_dropout,
        attention_probs_dropout_prob=attention_dropout
    )
    
    # Resize token embeddings to account for new special tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # Prepare datasets
    train_dataset = prepare_dataset(train_df, tokenizer, augment=True)
    valid_dataset = prepare_dataset(valid_df, tokenizer, augment=False)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./results/trial_{trial.number}",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        logging_dir=f"./logs/trial_{trial.number}",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=100,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_positive",
        greater_is_better=True,
        seed=42,
        fp16=False,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        warmup_ratio=warmup_ratio,
        gradient_accumulation_steps=2,
        label_smoothing_factor=label_smoothing,
        lr_scheduler_type="cosine_with_restarts"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    trainer.train()
    
    # Evaluate on validation set
    eval_results = trainer.evaluate(eval_dataset=valid_dataset)
    
    # Return validation F1 score as the objective value
    return eval_results["eval_f1_positive"]

def main():
    """Run hyperparameter optimization with Optuna"""
    # Create study with TPE sampler for more efficient search
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    
    # Run optimization with fewer trials
    study.optimize(objective, n_trials=5)
    
    # Print best trial
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Plot optimization history
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image("optimization_history.png")
    
    # Plot parameter importance
    fig = optuna.visualization.plotparam_importances(study)
    fig.write_image("param_importance.png")
    
    # Save best parameters to file
    with open("best_hyperparameters.txt", "w") as f:
        f.write("Best hyperparameters:\n")
        for key, value in trial.params.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nBest validation F1 score: {trial.value}")

if __name__ == "__main__":
    main() 