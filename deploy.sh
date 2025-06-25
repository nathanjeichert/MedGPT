#!/bin/bash

# MedGPT Cloud Run Deployment Script
# This script deploys the Medical Records Summary Tool to Google Cloud Run

set -e  # Exit on any error

# Configuration
PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-""}
REGION=${REGION:-"us-central1"}
SERVICE_NAME="medgpt"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🏥 MedGPT Cloud Run Deployment${NC}"
echo "=================================="

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}❌ Error: GOOGLE_CLOUD_PROJECT environment variable not set${NC}"
    echo "Please run: export GOOGLE_CLOUD_PROJECT=your-project-id"
    exit 1
fi

if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ Error: gcloud CLI not found${NC}"
    echo "Please install Google Cloud SDK: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Error: Docker not found${NC}"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites check passed${NC}"

# Authenticate and set project
echo -e "${YELLOW}Setting up Google Cloud project...${NC}"
gcloud config set project $PROJECT_ID

# Enable required APIs
echo -e "${YELLOW}Enabling required Google Cloud APIs...${NC}"
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build and deploy using Cloud Build
echo -e "${YELLOW}Starting Cloud Build deployment...${NC}"
gcloud builds submit --config cloudbuild.yaml

echo -e "${GREEN}🚀 Deployment completed successfully!${NC}"
echo ""
echo "Your MedGPT application should now be deployed to Cloud Run."
echo ""
echo "To get the service URL, run:"
echo "  gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)'"
echo ""
echo "To set up environment variables (like OPENAI_API_KEY), run:"
echo "  gcloud run services update $SERVICE_NAME --region=$REGION --set-env-vars OPENAI_API_KEY=your-key-here"
echo ""
echo -e "${YELLOW}⚠️  Important: Don't forget to set your OPENAI_API_KEY environment variable!${NC}"