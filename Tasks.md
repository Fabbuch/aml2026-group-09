Tasks
	- Label Distribution Sampler
	- Training Loop (Leo)
  - Evaluation loop (Vali)
  - Implementation of PoPe in vision transformer (Mattia)
  - Hperparameter tuning (look for default parameters…) (Fabio)
    
Workflow
  1. Read paper
	2. Understand and Inspect Dataset
	3. Recreate Data Splits
	4. Image augmentation pytorch
	5. Label Distribution Sampler
		- Compute Probabilities from training set
		- Predict randomly based on probabilities
		- Compute AUROC
	6. Vision Transformer Baseline
	7. Evaluation
		- Use prediction probabilities
		- AUROC per cancer type
		- Average AUROC
	8. Train with default Hyperparamenters
		- Does loss decrease?
		- Is AUROC better than Baseline?
	9. Hyperparameter tuning
		- Define search space
		- Train model and evaluate on validation set for each combination (AUROC)
		- Pick model with highes AUROC
	10. Final Evaluation on Test set
		- AUROC per cancer type
		- Average AUROC


