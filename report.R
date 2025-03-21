required_packages <- c(
  "tidyverse",     # For data manipulation and visualization
  "skimr",         # For data summary
  "mlr3",          # Primary ML framework
  "mlr3learners",  # For ML models
  "data.table",    # Used with mlr3
  "GGally",        # For visualization
  "MASS",          # For LDA/QDA
  "e1071",         # For SVM
  "rpart",         # For CART
  "ranger",        # For Random Forest
  "pROC",          # For ROC curves
  "reshape2",      # For data reshaping
  "gridExtra"      # For arranging multiple plots
)

# Install missing packages
new_packages <- 
  required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

# Load packages
library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(purrr)
library(tibble)
library(stringr)
library(forcats)
invisible(
  lapply(
    setdiff(required_packages, c("tidyverse")), library, character.only = TRUE))

# Set seed
set.seed(42)

loan_data <- read.csv("Personal Loan Data.csv")

str(loan_data)
summary(loan_data)

# Use skim to get summary statistics
skim(loan_data)

cat("\nPersonal Loan acceptance rate:",
    round(mean(loan_data$Personal.Loan == 1) * 100, 2), "%\n")

# Convert relevant variables to factors
loan_data$Personal.Loan <- as.factor(loan_data$Personal.Loan)
loan_data$Securities.Account <- as.factor(loan_data$Securities.Account)
loan_data$CD.Account <- as.factor(loan_data$CD.Account)
loan_data$Online <- as.factor(loan_data$Online)
loan_data$CreditCard <- as.factor(loan_data$CreditCard)
loan_data$Education <- factor(loan_data$Education,
                              levels = c(1, 2, 3),
                              labels = c("Undergraduate", 
                                         "Graduate", 
                                         "Advanced/Professional"))

# Use GGally examining relationship
key_vars_plot <- ggpairs(
  loan_data %>%
    dplyr::select(Personal.Loan, Income, CCAvg, Education, CD.Account),
  aes(color = Personal.Loan)
)
print(key_vars_plot)
ggsave("key_variables_relationships.png", key_vars_plot, width = 10, height = 8)

# Income by loan status
income_plot <- ggplot(
  loan_data, aes(x = Personal.Loan, y = Income, fill = Personal.Loan)) +
  geom_boxplot() +
  labs(title = "Income Distribution by Loan Status",
       x = "Personal Loan Accepted",
       y = "Income ($000s)") +
  theme_minimal()
print(income_plot)
ggsave("income_by_loan.png", income_plot, width = 8, height = 6)

# Education and loan status
edu_plot <- ggplot(loan_data, aes(x = Education, fill = Personal.Loan)) +
  geom_bar(position = "fill") +
  labs(title = "Loan Acceptance Rate by Education Level",
       x = "Education Level",
       y = "Proportion") +
  scale_y_continuous(labels = scales::percent) +
  theme_minimal()
print(edu_plot)
ggsave("education_by_loan.png", edu_plot, width = 8, height = 6)

# CD Account and loan status
cd_plot <- ggplot(loan_data, aes(x = CD.Account, fill = Personal.Loan)) +
  geom_bar(position = "fill") +
  labs(title = "Loan Acceptance Rate by CD Account Status",
       x = "Has CD Account",
       y = "Proportion") +
  scale_y_continuous(labels = scales::percent) +
  theme_minimal()
print(cd_plot)
ggsave("cd_account_by_loan.png", cd_plot, width = 8, height = 6)

# income versus loan status (density)
income_loan_plot <- ggplot(loan_data, aes(x = Income, fill = Personal.Loan)) +
  geom_density(alpha = 0.7) +
  labs(title = "Income Distribution by Loan Status",
       x = "Income ($000s)",
       y = "Density") +
  theme_minimal()
print(income_loan_plot)
ggsave("income_loan_density.png", income_loan_plot, width = 6, height = 4)

# income versus credit card spending
income_ccavg_plot <- ggplot(
  loan_data, aes(x = Income, y = CCAvg, color = Personal.Loan)) +
  geom_point(alpha = 0.5) +
  labs(title = "Income vs Credit Card Spending",
       x = "Income ($000s)",
       y = "Credit Card Avg Spending ($000s)") +
  theme_minimal()
print(income_ccavg_plot)
ggsave("income_ccavg_plot.png", income_ccavg_plot, width = 6, height = 4)

# key relationships in personal loan data (combination)
description_plots <- list(income_loan_plot, cd_plot, edu_plot)
combined_description <- gridExtra::grid.arrange(
  grobs = description_plots,
  ncol = 3,
  top = "Key Relationships in Personal Loan Data"
)
ggsave("combined_description_plots.png", 
       combined_description, width = 12, height = 4)

# Create feature for Income per family member
loan_data$Income_per_Family <- loan_data$Income / loan_data$Family

# Remove ZIP.Code
loan_data$ZIP.Code <- NULL
loan_data$Personal.Loan <- as.factor(loan_data$Personal.Loan)

# Define task
loan_task <- TaskClassif$new(
  id = "loan_prediction",
  backend = loan_data,
  target = "Personal.Loan",
  positive = "1"
)

# Create train/test split
set.seed(123)
split <- partition(loan_task, ratio = 0.7)
train_set <- split$train
test_set <- split$test

# Define 5-fold cross-validation
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(loan_task)

# Logistic Regression
learner_logreg <- lrn("classif.log_reg", predict_type = "prob")

# Linear Discriminant Analysis
learner_lda <- lrn("classif.lda", predict_type = "prob")

# Quadratic Discriminant Analysis
learner_qda <- lrn("classif.qda", predict_type = "prob")

# CART Decision Tree
learner_cart <- lrn("classif.rpart", predict_type = "prob")

# Random Forest
learner_rf <- lrn("classif.ranger", 
                  predict_type = "prob", importance = "permutation")

# Baseline Classifier
learner_baseline <- lrn("classif.featureless", predict_type = "prob")

# Learners for benchmarking
learners <- list(
  learner_baseline,
  learner_logreg,
  learner_lda,
  learner_qda,
  learner_cart,
  learner_rf
)

# Define benchmark design
benchmark_design <- benchmark_grid(
  tasks = loan_task,
  learners = learners,
  resamplings = cv5
)

benchmark_result <- benchmark(benchmark_design)

# Calculate different performance metrics
performance_metrics <- benchmark_result$aggregate(c(
  msr("classif.ce"),       # Classification error
  msr("classif.acc"),      # Accuracy
  msr("classif.auc"),      # Area under ROC curve
  msr("classif.fpr"),      # False positive rate
  msr("classif.fnr")       # False negative rate
))

# Print model comparison
print(performance_metrics[order(performance_metrics$classif.auc, 
                                decreasing = TRUE),])

# Create a plot comparing model performance
model_comparison_data <- as.data.frame(performance_metrics)
model_comparison_data$learner_id <- 
  factor(model_comparison_data$learner_id, 
         levels = model_comparison_data$learner_id[
           order(model_comparison_data$classif.auc, decreasing = TRUE)])

# Plot AUC comparison
model_comparison_plot <- ggplot(
  model_comparison_data, 
  aes(x = learner_id, y = classif.auc, fill = learner_id)) +
  geom_bar(stat = "identity") +
  labs(title = "Model Comparison - AUC",
       x = "Model",
       y = "AUC Score") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none")
print(model_comparison_plot)
ggsave("model_comparison_auc.png", model_comparison_plot, width = 8, height = 5)

# Create improved model with SMOTE or ROSE
if ("mlr3pipelines" %in% installed.packages() && 
    "mlr3filters" %in% installed.packages()) {
  
  library(mlr3pipelines)
  library(mlr3filters)
  
  if (requireNamespace("mlr3extralearners", quietly = TRUE)) {
    
    library(mlr3extralearners)
    if ("classif.smote" %in% mlr_learners$keys()) {
      po_smote <- po("smote")
      rf_base <- lrn("classif.ranger", 
                     predict_type = "prob", 
                     importance = "permutation")
      rf_pipe <- po_smote %>>% rf_base
      rf_balanced_learner <- as_learner(rf_pipe)
    } else {
      rf_balanced_learner <- lrn("classif.ranger",
                                 predict_type = "prob",
                                 importance = "permutation",
                                 class.weights = c("0" = 1, "1" = 9))
    }
  } else {
    rf_balanced_learner <- lrn("classif.ranger",
                               predict_type = "prob",
                               importance = "permutation",
                               class.weights = c("0" = 1, "1" = 9))
  }
} else {
  rf_balanced_learner <- lrn("classif.ranger",
                             predict_type = "prob",
                             importance = "permutation",
                             class.weights = c("0" = 1, "1" = 9))
}

# Compare base model vs. balanced model
learners_balanced <- list(
  lrn("classif.ranger", predict_type = "prob", id = "rf_base"),
  rf_balanced_learner
)

balanced_design <- benchmark_grid(
  tasks = loan_task,
  learners = learners_balanced,
  resamplings = cv5
)

balanced_results <- benchmark(balanced_design)
balanced_performance <- balanced_results$aggregate(c(
  msr("classif.ce"),
  msr("classif.acc"),
  msr("classif.auc")
))

print(balanced_performance)

# Determine best model
best_model <- rf_balanced_learner
best_model$train(loan_task, row_ids = train_set)

# Make predictions on test set
predictions <- best_model$predict(loan_task, row_ids = test_set)

# Print confusion matrix
print(predictions$confusion)

# Calculate and display ROC curve data
roc_data <- predictions$score(msr("classif.auc"))
print(paste("Test AUC:", round(roc_data, 4)))

# Create ROC curve for the best model
probs <- predictions$prob[, "1"]
actual <- factor(loan_data$Personal.Loan[test_set], levels = c("0", "1"))

# Create ROC object and plot
roc_obj <- roc(as.numeric(actual) - 1, probs)
roc_plot <- ggroc(roc_obj) +
  geom_abline(intercept = 1, slope = 1, linetype = "dashed", alpha = 0.7) +
  labs(title = "ROC Curve for Personal Loan Prediction",
       subtitle = paste("AUC =", round(auc(roc_obj), 3)),
       x = "False Positive Rate (1 - Specificity)", 
       y = "True Positive Rate (Sensitivity)") +
  theme_minimal()
print(roc_plot)
ggsave("roc_curve.png", roc_plot, width = 6, height = 5)

# Get variable importance
if ("importance" %in% names(best_model$param_set$values)) {
  if (inherits(best_model$model, "ranger") && 
      !is.null(best_model$model$variable.importance)) {
    importance_scores <- best_model$model$variable.importance
    importance_df <- data.frame(
      Feature = names(importance_scores),
      Importance = importance_scores
    )
    importance_df <- importance_df[order(
      importance_df$Importance, decreasing = TRUE), ]
    
    print("Top 10 most important features:")
    print(head(importance_df, 10))
    
    importance_plot <- ggplot(
      head(importance_df, 10), 
      aes(x = reorder(Feature, Importance), y = Importance)) +
      geom_col() +
      coord_flip() +
      labs(title = "Top 10 Feature Importance", 
           x = "Feature", y = "Importance") +
      theme_minimal()
    
    print(importance_plot)
    ggsave("feature_importance.png", importance_plot, width = 8, height = 6)
  }
}

# Try different thresholds
thresholds <- seq(0.1, 0.9, by = 0.05)
results <- data.frame(threshold = thresholds, 
                      precision = NA, 
                      recall = NA, 
                      f1 = NA)

for (i in seq_along(thresholds)) {
  pred_class <- factor(ifelse(probs >= thresholds[i], "1", "0"), 
                       levels = c("0", "1"))
  
  cm <- table(Actual = actual, Predicted = pred_class)
  precision <- ifelse(sum(pred_class == "1") > 0,
                      cm[2,2] / sum(pred_class == "1"), 0)
  
  recall <- ifelse(sum(actual == "1") > 0,
                   cm[2,2] / sum(actual == "1"), 0)
  
  f1 <- ifelse(precision > 0 & recall > 0,
               2 * precision * recall / (precision + recall), 0)
  
  results$precision[i] <- precision
  results$recall[i] <- recall
  results$f1[i] <- f1
}

# Find optimal threshold
opt_threshold <- results$threshold[which.max(results$f1)]

# Plot threshold vs F1 score
threshold_plot <- ggplot(results, aes(x = threshold)) +
  geom_line(aes(y = precision, color = "Precision")) +
  geom_line(aes(y = recall, color = "Recall")) +
  geom_line(aes(y = f1, color = "F1 Score")) +
  geom_vline(xintercept = opt_threshold, linetype = "dashed") +
  labs(title = "Performance Metrics vs Threshold",
       x = "Threshold",
       y = "Score",
       color = "Metric") +
  theme_minimal()

print(threshold_plot)
ggsave("threshold_optimization.png", threshold_plot, width = 8, height = 6)

# Create final predictions with optimal threshold
final_preds <- factor(ifelse(probs >= opt_threshold, "1", "0"),
                      levels = c("0", "1"))

if (is.factor(actual) && is.factor(final_preds)) {
  final_cm <- table(Actual = actual, Predicted = final_preds)
} else {
  final_cm <- table(Actual = as.character(actual), 
                    Predicted = as.character(final_preds))
}

print("Final confusion matrix with optimal threshold:")
print(final_cm)

# Create heatmap
if (exists("final_cm") && 
    is.table(final_cm) && 
    nrow(final_cm) == 2 && 
    ncol(final_cm) == 2) {
  
  confusion_data <- melt(final_cm)
  names(confusion_data) <- c("Actual", "Predicted", "Value")
  
  confusion_data$Percentage <- 
    paste0(round(confusion_data$Value / sum(final_cm) * 100, 1), "%")
  
  cm_heatmap <- ggplot(
    confusion_data, 
    aes(x = Predicted, y = Actual, fill = Value)) +
    geom_tile() +
    geom_text(aes(label = Value), color = "white", size = 6) +
    geom_text(aes(label = Percentage), color = "white", size = 4, vjust = 3) +
    scale_fill_gradient(low = "steelblue", high = "darkblue") +
    labs(title = "Confusion Matrix",
         subtitle = paste("Threshold =", opt_threshold)) +
    theme_minimal() +
    theme(legend.position = "none")
  
  print(cm_heatmap)
  ggsave("confusion_matrix_heatmap.png", cm_heatmap, width = 6, height = 5)
}

# Calculate final metrics with robust handling of edge cases
if (exists("final_cm") &&
    is.table(final_cm) && 
    nrow(final_cm) == 2 && 
    ncol(final_cm) == 2) {
  tp <- final_cm[2,2]
  fp <- final_cm[1,2]
  fn <- final_cm[2,1]
  tn <- final_cm[1,1]
  
  final_accuracy <- (tp + tn) / sum(final_cm)
  final_precision <- ifelse(tp + fp > 0, tp / (tp + fp), 0)
  final_recall <- ifelse(tp + fn > 0, tp / (tp + fn), 0)
  final_f1 <- ifelse(
    final_precision > 0 & final_recall > 0,
    2 * final_precision * final_recall / (final_precision + final_recall), 0)
  
  performance_summary <- data.frame(
    Metric = c("Accuracy", "Precision", "Recall", "F1 Score", "AUC"),
    Value = c(round(final_accuracy, 3), 
              round(final_precision, 3),
              round(final_recall, 3),
              round(final_f1, 3),
              round(auc(roc_obj), 3))
  )
  
  write.csv(performance_summary, "performance_summary.csv", row.names = FALSE)
  
  profit_matrix <- matrix(c(0, -800, -200, 1000), nrow = 2)
  rownames(profit_matrix) <- colnames(profit_matrix) <- c("0", "1")
  
  total_profit <- tn * profit_matrix[1,1] +
    fp * profit_matrix[1,2] +
    fn * profit_matrix[2,1] +
    tp * profit_matrix[2,2]
  
  profit_per_customer <- total_profit / length(test_set)
  
  # Calculate profit for different thresholds
  profit_by_threshold <- data.frame(
    threshold = thresholds,
    profit = NA
  )
  
  for (i in seq_along(thresholds)) {
    pred_class <- factor(ifelse(probs >= thresholds[i], "1", "0"), 
                         levels = c("0", "1"))
    cm <- table(Actual = actual, Predicted = pred_class)
    
    if (nrow(cm) == 2 && ncol(cm) == 2) {
      tn_i <- cm[1,1]
      fp_i <- cm[1,2]
      fn_i <- cm[2,1] 
      tp_i <- cm[2,2]
      
      profit <- tn_i * profit_matrix[1,1] +
        fp_i * profit_matrix[1,2] +
        fn_i * profit_matrix[2,1] +
        tp_i * profit_matrix[2,2]
      
      profit_by_threshold$profit[i] <- profit
    }
  }
  
  # Plot profit by threshold
  profit_plot <- ggplot(profit_by_threshold, aes(x = threshold, y = profit)) +
    geom_line(color = "darkgreen", size = 1.2) +
    geom_point() +
    geom_vline(xintercept = opt_threshold, linetype = "dashed", color = "red") +
    labs(title = "Profit by Classification Threshold",
         subtitle = "Vertical line shows F1-optimal threshold",
         x = "Threshold",
         y = "Profit ($)") +
    theme_minimal()
  
  print(profit_plot)
  ggsave("profit_by_threshold.png", profit_plot, width = 7, height = 5)
}

# Create a combined plot
if (exists("importance_df") && exists("results") && exists("final_cm")) {
  # Focus on top 3 predictors from random forest
  top_predictors <- head(importance_df$Feature, 3)
  
  if ("Income" %in% top_predictors) {
    loan_data$Income_Bracket <- 
      cut(loan_data$Income, 
          breaks = c(0, 50, 100, 150, Inf),
          labels = c("<50K", "50K-100K", "100K-150K", ">150K"))
    
    income_bracket_plot <- 
      ggplot(
        loan_data, aes(x = Income_Bracket, fill = Personal.Loan)) +
      geom_bar(position = "fill") +
      labs(title = "Loan Acceptance Rate by Income Bracket",
           x = "Income Bracket",
           y = "Proportion") +
      scale_y_continuous(labels = scales::percent) +
      theme_minimal()
    
    ggsave("income_bracket_acceptance.png", 
           income_bracket_plot, width = 6, height = 4)
  }
  
  #if ("Education" %in% top_predictors) { }
  
  summary_plots <- list()
  
  if (exists("income_plot")) summary_plots[[1]] <- income_plot
  
  if (exists("roc_plot")) summary_plots[[2]] <- roc_plot
  
  if (exists("cm_heatmap")) summary_plots[[3]] <- cm_heatmap
  
  if (length(summary_plots) > 0) {
    executive_summary <- gridExtra::grid.arrange(
      grobs = summary_plots,
      ncol = 2,
      top = "Personal Loan Classification: Key Results"
    )
    ggsave("executive_summary_dashboard.png", 
           executive_summary, width = 10, height = 8)
  }
}