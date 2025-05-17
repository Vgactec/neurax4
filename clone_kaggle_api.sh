#!/bin/bash

# Script to clone the Kaggle API repository and remove workflow files
# This prevents any CI/CD loops while preserving the code for study

# Set up error handling
set -e
echo "Starting Kaggle API repository cloning process..."

# Clone the repository
echo "Cloning the Kaggle API repository..."
git clone https://github.com/Kaggle/kaggle-api.git

# Navigate into the repository
cd kaggle-api

# Remove workflow files to prevent CI/CD loops
echo "Removing workflow files to prevent execution loops..."
if [ -d ".github/workflows" ]; then
    rm -rf .github/workflows
    echo "Workflow files removed successfully."
else
    echo "No workflow directory found at .github/workflows"
fi

# Check for any other CI configuration files and remove them
if [ -f ".travis.yml" ]; then
    rm .travis.yml
    echo "Travis CI configuration removed."
fi

if [ -f ".gitlab-ci.yml" ]; then
    rm .gitlab-ci.yml
    echo "GitLab CI configuration removed."
fi

if [ -f "azure-pipelines.yml" ]; then
    rm azure-pipelines.yml
    echo "Azure Pipelines configuration removed."
fi

# Return to the original directory
cd ..

echo "Process completed successfully."
echo "The Kaggle API repository has been cloned and all workflow files have been removed."
echo "The code is now ready for exploration and study with no risk of automated processes running."
echo "Repository location: ./kaggle-api"
