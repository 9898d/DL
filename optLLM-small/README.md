---
license: apache-2.0
base_model: google/flan-t5-small
tags:
- generated_from_trainer
model-index:
- name: optLLM-small
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

## Model Overview

This model is a fine-tuned version of the `google/flan-t5-small` on a synthetic dataset designed for supply chain optimization problems. 
The primary goal of this model is to classify the type of supply chain optimization problem (e.g., Newsvendor, EOQ, VRP) described in the input and extract the relevant variables necessary for solving these problems.
The idea is that anyone, even with no background in optimisation , can interact with a solver and solve his problem: for example imagine a salesperson in a store with no experience in optimization, he could describe their problem in natural language and obtain an optimal solution

## Model Details

- **Model Name**: Supply Chain Optimization LLM
- **Base Model**: [google/flan-t5-small](https://huggingface.co/google/flan-t5-small)
- **Fine-tuning Data**: Synthetic dataset consisting of supply chain problem descriptions with corresponding problem types and variables.
- **Framework**: PyTorch
- **Tokenizer**: AutoTokenizer from `google/flan-t5-small`
- It achieves the following results on the evaluation set:
- Loss: 0.0046



## Model description

### Use Cases

This model is designed to be used in the following scenarios:
1. **Classification**: Identifying the type of supply chain optimization problem from a textual description.
2. **Variable Extraction**: Extracting relevant variables needed for solving the identified optimization problem.

### Supported Problem Types

The model can classify and extract variables for the following supply chain optimization problems:
1. **Newsvendor Problem**: Determines the optimal inventory level to maximize expected profit.
2. **Economic Order Quantity (EOQ)**: Calculates the optimal order quantity to minimize total inventory costs.
3. **Vehicle Routing Problem (VRP)**: Plans optimal routes for a fleet of vehicles to deliver products to customers.


### Example Inputs

- **Newsvendor Problem**: "We have 12 items of X and 7 items of Y left in our inventory. The cost per item of X is $5 and of Y is $15. The storage cost is $1 per unit. We have historical orders for X: [26, 25, 29, 17, 49, 33, 46, 42, 30, 33] and for Y: [31, 26, 50, 25, 21, 16, 21, 39, 27, 45]. The selling price for X is $3 and for Y is $3. The storage limit is 498 units."
- **EOQ Problem**: "We need to determine the optimal order quantity for a product with a demand rate of 500 units per year, an ordering cost of $50 per order, and a holding cost of $2 per unit per year."
- **VRP Problem**: "We need to plan routes for 3 vehicles to deliver products to 10 customers.
Each vehicle has a capacity of 100 units. The customer demands are as follows: 10, 20, 15, 10, 5, 25, 30, 10, 15, 20 units.
The coordinates of the delivery locations are: (2,3), (5,5), (8,8), (1,2), (3,6), (7,2), (6,6), (5,9), (2,8), (4,4).
The warehouse is located at (0,0)."


## Fine-Tuning Process

The model was fine-tuned using a synthetic dataset with the following steps:
1. **Dataset Generation**: Synthetic data was generated for each problem type with specific formats and variables.
2. **Training**: The `google/flan-t5-small` model was fine-tuned on this dataset, with a focus on accurately classifying the problem type and extracting relevant variables.

## Performance

The model has shown good performance on the synthetic dataset, with a high accuracy in classifying problem types and extracting variables. However, performance on real-world data may vary, and further fine-tuning or validation on real-world datasets is recommended.

## Limitations

- The model is primarily trained on synthetic data and may require further fine-tuning for specific real-world applications.
- Handling of extremely large numbers of variables (e.g., hundreds of products) may need optimization.
## Future work 
I  tried to make the model dynamic and agnostic the number of params/variables in the problem , however due to a bug I’m still trying to identify the model always ignores the last product in the newsvendor problem (open to ideas if someone has a clue why is this happening ) 

Next steps would be fixing the bug with the newsvendor problem , and developing the routing interface to communicate with the appropriate solvers to find an optimal solution for the user 



### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 4
- eval_batch_size: 4
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 1

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| 0.0161        | 1.0   | 4000 | 0.0046          |


### Framework versions

- Transformers 4.41.2
- Pytorch 2.3.0+cu121
- Datasets 2.20.0
- Tokenizers 0.19.1
