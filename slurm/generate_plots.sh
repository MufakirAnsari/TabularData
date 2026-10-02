#!/bin/bash
# Master Plotting Dispatcher
# Run this interactively AFTER the evaluation jobs have completed!
# It will generate all PCA and TSTR plots for all 6 datasets.

DATASETS=("ALLAML" "CLL_SUB_111" "colon" "Prostate_GE" "SMK_CAN_187" "GLI_85")

export TABULAR_ROOT=$HOME/TabularData
cd $TABULAR_ROOT
source venv/bin/activate

for DATASET in "${DATASETS[@]}"; do
    echo "======================================================="
    echo "Generating Plots for Dataset: $DATASET"
    echo "======================================================="
    
    echo "  -> Running PCA Overlap Generation..."
    python -m src.pca_evaluation --dataset $DATASET
    
    echo "  -> Running TSTR Visualizations..."
    python -m src.plot_tstr --dataset $DATASET
    
    echo "  Done with $DATASET!"
    echo ""
done

echo "All plotting complete! Check results/<dataset>/pca_plots/ and results/<dataset>/tstr_plots/"
