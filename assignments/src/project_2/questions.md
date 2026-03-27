Based on a thorough review of your report and notebooks against the project requirements, your work is technically sound and covers all mandatory bases. However, to make it truly "thorough" and ensure you don't lose points on specific descriptive requirements, there are a few areas you should expand upon.
1. Object Localization: Missing Descriptive Depth

The project description requires you to discuss your choice of architecture and loss functions in detail.

    Architecture Detail: In your report, you state that the ResNet model worked best. To be more thorough, you should describe the modifications you made to the standard ResNet architecture. Specifically, mention in your report that you replaced the final fully connected layer to output a vector of size 15 (1 for object presence, 4 for coordinates, and 10 for class probabilities) as seen in your code.

    Loss Function Weighting: In your Object_localization.ipynb, you used a weighted loss: loss = 10 * loss_obj + loss_box + loss_class. This is a significant design choice—it shows you recognized that detecting the presence of an object is more critical (or harder to learn) than the other tasks. You should explain this weighting in the "Loss Function" section of your report to demonstrate technical depth.

    Failure Analysis: The requirements ask for potential reasons why results might differ or code might not work. While you mention overfitting, you could be more specific. For example, mention if your model struggled with specific digits (e.g., confusing '1' and '7') or if the "de-noising" threshold of 1.5 occasionally removed parts of the actual digits.

2. Object Detection: Missing Technical Explanations

This section has specific descriptive requirements that are currently a bit thin in your report.

    Transformation Logic: The project requires you to "describe how the ground truth is transformed into grid labels". Your code contains the global_to_local function which handles this , but your report only mentions that you used a 2x3 grid. You should add a sentence or two explaining the math: how a global coordinate is mapped to a specific cell and then converted into a coordinate relative to that cell’s top-left corner.

    Model Output Shape: You are required to describe the output shape of your model. You should explicitly state in the report that your model outputs a tensor (or flattened vector) corresponding to 2×3×7 values (where 7 represents: 1 confidence score, 4 coordinates, and 2 class probabilities for digits 0 and 1).

    NMS Explanation: You mention that NMS didn't affect your best model much , but the instructions require you to "describe how it works and what parameters you use". You should explicitly define NMS in your report: explain that it removes overlapping boxes by comparing their IoU against a threshold (0.5 in your code) and keeping only the one with the highest confidence score.

3. Checklist of Mandatory Deliverables

    Division of Labor: Included and clear.

    AI Disclosure: Included and follows UiB guidelines.

    Training/Validation Loss Plots: Mentioned in the report and implemented in the notebooks. Note: Ensure these image files (e.g., loss_over_time_resnet.png) are actually included in your final .zip submission folder, as they are referenced but not embedded as text.

    Test Set Evaluation: You have performed this in your notebooks. Ensure the final metrics (IoU, Accuracy, mAP) are clearly listed in a summary table in your report for easy reading by the grader.

Summary of Recommendations

    Add the mathematical logic for the grid transformation (Global → Local) in the Detection section.

    Explicitly state the output shapes of both models (e.g., "The model outputs a vector of size 15 for localization...").

    Define NMS and your chosen IoU threshold (0.5) in the text.

    Discuss the 10x loss weighting you used in the localization task to show you fine-tuned the training process.