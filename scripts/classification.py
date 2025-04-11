import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import pandas as pd
from datasets import Dataset
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score
import os
import shutil
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_dataset(df, tokenizer, max_length=128, augment=False):
    """Prepare dataset for training VAEM classification
    
    The model should focus on identifying vaccine adverse events mentions (VAEM) in the comment part,
    while using the title (if present) as context. For posts with title:comment format,
    the prediction should be based primarily on the comment part.
    
    Args:
        df: DataFrame containing the data
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        augment: Whether to apply data augmentation
    """
    def tokenize_function(examples):
        texts = []
        for title, content in zip(examples['title'], examples['cleaned_text']):
            if title and title.strip():
                # Add title with a special prefix but give it less prominence
                # Use [CONTEXT] for title to indicate it's supporting information
                # Use [POST] for the main content to emphasize it's the primary text for prediction
                combined_text = f"[CONTEXT] {title} [POST] {content}"
            else:
                # If no title, just use the content with [POST] marker
                combined_text = f"[POST] {content}"
            
            # Apply simple data augmentation if enabled
            if augment and np.random.random() < 0.5:
                # Randomly remove some words (simplified version of dropout)
                words = combined_text.split()
                if len(words) > 10:  # Only augment if there are enough words
                    keep_prob = 0.9  # Keep 90% of words
                    words = [w for w in words if np.random.random() < keep_prob]
                    combined_text = ' '.join(words)
            
            texts.append(combined_text)
        
        # Tokenize with truncation from the left to ensure we keep the end of long posts
        # This is important as the comment part (which comes after the title) is more crucial
        return tokenizer(texts, 
                        padding='max_length',
                        truncation=True,
                        max_length=max_length,
                        padding_side='right',  # Default padding on the right
                        stride=0,
                        return_overflowing_tokens=False)
    
    print("Converting DataFrame to Dataset...")
    dataset = Dataset.from_pandas(df)
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function, 
                                  batched=True)
    return tokenized_dataset

def compute_metrics(pred):
    """Compute metrics for evaluation"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Calculate metrics for both classes
    precision_both, recall_both, f1_both, support = precision_recall_fscore_support(labels, preds, average=None, labels=[0, 1])
    
    # Calculate macro average (average of per-class metrics)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average='macro')
    
    # Calculate metrics for positive class (binary average automatically uses label 1 as positive)
    precision_pos, recall_pos, f1_pos, _ = precision_recall_fscore_support(labels, preds, average='binary')
    
    acc = accuracy_score(labels, preds)
    
    # Create detailed metrics dictionary
    return {
        'accuracy': acc,
        'f1_macro': f1_macro,          # Average of both classes' F1 scores
        'f1_negative': f1_both[0],     # Class 0 (non-adverse effects)
        'f1_positive': f1_pos,         # Class 1 (adverse effects) using binary average
        'precision_macro': precision_macro,
        'precision_negative': precision_both[0],
        'precision_positive': precision_pos,
        'recall_macro': recall_macro,
        'recall_negative': recall_both[0],
        'recall_positive': recall_pos,
        'support_negative': int(support[0]),  # Number of samples in class 0
        'support_positive': int(support[1])   # Number of samples in class 1
    }

def save_model_safely(model, tokenizer, final_path):
    """Save model and tokenizer safely using a temporary directory"""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Saving model to temporary directory: {temp_dir}")
        # Save to temporary directory first
        model.save_pretrained(temp_dir, safe_serialization=True)
        tokenizer.save_pretrained(temp_dir)
        
        # Create the final directory if it doesn't exist
        os.makedirs(final_path, exist_ok=True)
        
        # Remove the final directory if it exists
        if os.path.exists(final_path):
            shutil.rmtree(final_path)
        
        # Move from temporary to final location
        shutil.copytree(temp_dir, final_path)
        print(f"Model successfully saved to {final_path}")

def generate_predictions(model, tokenizer, test_df, output_file="prediction_task6.csv"):
    """Generate predictions for test data and save to CSV"""
    print("Generating predictions...")
    
    # Prepare test dataset
    test_dataset = prepare_dataset(test_df, tokenizer)
    
    # Get predictions
    trainer = Trainer(model=model)
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    
    # Create output DataFrame
    output_df = pd.DataFrame({
        'id': test_df['id'],
        'labels': pred_labels
    })
    
    # Save to CSV
    output_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    
    return output_df

def plot_training_metrics(trainer):
    """Plot training and validation metrics over time"""
    # Extract metrics from training history
    history = trainer.state.log_history
    
    # Separate training and validation metrics
    train_metrics = [(x['step'], x['loss']) for x in history if 'loss' in x and 'eval_loss' not in x]
    eval_metrics = [(x['step'], x['eval_loss'], x['eval_f1_positive'], x['eval_f1_macro'], x['eval_accuracy']) 
                   for x in history if 'eval_loss' in x]
    
    # Create figure with three subplots (removed class distribution plot as it's not needed)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    
    # Plot training and validation loss
    if train_metrics:
        steps, losses = zip(*train_metrics)
        ax1.plot(steps, losses, label='Training Loss', marker='o')
    if eval_metrics:
        steps, eval_losses, _, _, _ = zip(*eval_metrics)
        ax1.plot(steps, eval_losses, label='Validation Loss', marker='s')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot validation F1 scores
    if eval_metrics:
        steps, _, f1_pos, f1_macro, _ = zip(*eval_metrics)
        ax2.plot(steps, f1_pos, label='F1 Positive Class', marker='o', color='green')
        ax2.plot(steps, f1_macro, label='F1 Macro Avg', marker='s', color='blue')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Validation F1 Scores')
    ax2.legend()
    ax2.grid(True)
    
    # Plot validation accuracy
    if eval_metrics:
        steps, _, _, _, accuracies = zip(*eval_metrics)
        ax3.plot(steps, accuracies, label='Accuracy', marker='o', color='purple')
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Validation Accuracy')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()
    print("Training metrics plot saved as training_metrics.png")

def main():
    print("Loading data...")
    # Load data
    train_df = pd.read_csv("processed_train.csv")
    valid_df = pd.read_csv("processed_valid.csv")
    test_df = pd.read_csv("processed_test.csv")
    
    print("Initializing model and tokenizer...")
    model_name = "cardiffnlp/twitter-roberta-large-2022-154m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add special tokens to tokenizer
    special_tokens = {'additional_special_tokens': ['[CONTEXT]', '[POST]']}
    num_added_tokens = tokenizer.add_special_tokens(special_tokens)
    
    # Initialize model with updated vocab size and dropout
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        ignore_mismatched_sizes=True,
        hidden_dropout_prob=0.2,  # Increased dropout
        attention_probs_dropout_prob=0.2  # Increased attention dropout
    )
    
    # Resize token embeddings to account for new special tokens
    model.resize_token_embeddings(len(tokenizer))
    
    print("Preparing datasets...")
    # Prepare datasets with data augmentation for training
    train_dataset = prepare_dataset(train_df, tokenizer, augment=True)
    valid_dataset = prepare_dataset(valid_df, tokenizer, augment=False)
    
    print("Setting up training arguments...")
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,  # Slightly higher learning rate
        per_device_train_batch_size=24,  # Larger batch size for better generalization
        per_device_eval_batch_size=24,
        num_train_epochs=4,  # Reduced epochs
        weight_decay=0.02,  # Increased weight decay
        logging_dir="./logs",
        logging_steps=30,
        eval_strategy="steps",
        eval_steps=15,
        save_steps=30,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_positive",
        greater_is_better=True,
        seed=42,
        fp16=False,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        warmup_ratio=0.1,  # Add warmup for learning rate
        gradient_accumulation_steps=2,  # Effective batch size = 24 * 2 = 48
        label_smoothing_factor=0.15,  # Increased label smoothing
        lr_scheduler_type="cosine_with_restarts"  # Added cosine learning rate scheduler with restarts
    )
    
    # print("Configuring LoRA...")
    # # Configure LoRA with adjusted parameters
    # peft_config = LoraConfig(
    #     task_type=TaskType.SEQ_CLS,
    #     inference_mode=False,
    #     r=16,  # Increased rank for more capacity
    #     lora_alpha=32,
    #     lora_dropout=0.2,  # Increased dropout for regularization
    #     target_modules=["query", "key", "value", "dense"],  # Removed 'output' as it's not a direct Linear layer
    #     modules_to_save=["embedding", "classifier"],
    #     bias="all"  # Enable bias training
    # )
    
    # # Get PEFT model
    # model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()
    
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("Starting training...")
    # Train the model
    train_result = trainer.train()
    
    print("Plotting training metrics...")
    plot_training_metrics(trainer)

    print("Saving model and tokenizer...")
    # Save model and tokenizer safely
    save_model_safely(model, tokenizer, "./final_model")
    
    print("Evaluating on training set...")
    # Evaluate on training set
    train_results = trainer.evaluate(eval_dataset=train_dataset)
    print(f"Training Results: {train_results}")
    
    print("Evaluating on validation set...")
    # Evaluate on validation set
    valid_results = trainer.evaluate(eval_dataset=valid_dataset)
    print(f"Validation Results: {valid_results}")
    
    # Save validation predictions
    print("Generating validation predictions...")
    valid_predictions = trainer.predict(valid_dataset)
    valid_pred_labels = np.argmax(valid_predictions.predictions, axis=1)
    
    # Create validation output DataFrame
    valid_output_df = pd.DataFrame({
        'id': valid_df['id'],
        'labels': valid_pred_labels,
        'true_labels': valid_df['labels']  # Include true labels for comparison
    })
    
    # Save to CSV
    valid_output_df.to_csv('validation_predictions.csv', index=False)
    print("Validation predictions saved to validation_predictions.csv")
    
    print("Generating predictions...")
    # Generate predictions using the current model directly
    # Instead of loading a saved model, use the current one
    generate_predictions(model, tokenizer, test_df)

if __name__ == "__main__":
    main() 
    