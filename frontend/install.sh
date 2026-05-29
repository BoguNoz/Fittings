#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Colors for better console readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== STARTING AUTOMATIC INSTALLATION AND DEPLOYMENT ===${NC}\n"

# 1. Update package lists
echo -e "${YELLOW}[1/5] Updating system package lists...${NC}"
sudo apt-get update -y

# 2. Install Git and basic tools
echo -e "${YELLOW}[2/5] Installing curl, git, and software properties...${NC}"
sudo apt-get install -y curl git software-properties-common

# 3. Check and install Docker and Docker Compose
echo -e "${YELLOW}[3/5] Checking and installing Docker...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${BLUE}Downloading and running the official Docker install script...${NC}"
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    rm get-docker.sh
else
    echo -e "${GREEN}Docker is already installed.${NC}"
fi

# Install Docker Compose plugin
echo -e "${BLUE}Installing Docker Compose plugin...${NC}"
sudo apt-get install -y docker-compose-plugin

# Add current user to docker group to avoid using sudo later
echo -e "${BLUE}Adding user $(whoami) to the docker group...${NC}"
sudo usermod -aG docker $USER

# 4. Clone the repository
REPO_DIR="Fittings"
echo -e "${YELLOW}[4/5] Cloning repository from GitHub...${NC}"
if [ -d "$REPO_DIR" ]; then
    echo -e "${YELLOW}Folder $REPO_DIR already exists. Updating the code...${NC}"
    cd $REPO_DIR && git pull && cd ..
else
    git clone https://github.com/BoguNoz/Fittings.git
fi

# 5. Build and run Docker containers
echo -e "${YELLOW}[5/5] Automatically building and launching the application...${NC}"

# NOTE: We use 'sudo docker' here because group changes take effect only after relog or running 'newgrp'
# This ensures the script won't fail due to permissions during the initial setup.

# A. Run Backend (Docker Compose)
echo -e "${BLUE}Building and launching BACKEND (unlimited resource access)...${NC}"
cd "$REPO_DIR/backend"
sudo docker compose up --build -d

# B. Build and run Frontend (Dockerfile)
echo -e "${BLUE}Building and launching FRONTEND...${NC}"
cd "../frontend"
sudo docker build -t fittings-frontend .
sudo docker run -d -p 80:80 --name ptr-frontend --restart unless-stopped fittings-frontend

# Return to the main directory
cd ../..

# Clear screen before showing user instructions
clear

echo -e "${GREEN}==================================================================${NC}"
echo -e "${GREEN}   SUCCESS! ENVIRONMENT HAS BEEN CONFIGURATED AND LAUNCHED        ${NC}"
echo -e "${GREEN}==================================================================${NC}"
echo -e ""
echo -e "${CYAN}Where to find the application?${NC}"
echo -e "  - ${YELLOW}FRONTEND:${NC} Open your browser and go to: http://localhost"
echo -e "  - ${YELLOW}BACKEND:${NC}  Running at:                    http://localhost:8000"
echo -e ""
echo -e "${CYAN}System Resources:${NC}"
echo -e "  - The containers have been launched without any artificial RAM/CPU limits."
echo -e "    The Backend will automatically use 100% of the available system resources as needed."
echo -e ""
echo -e "${CYAN}Important Configuration Step (Do this now!):${NC}"
echo -e "  To manage Docker in the future without typing 'sudo', type this command now:"
echo -e "  ${GREEN}newgrp docker${NC}"
echo -e ""
echo -e "${CYAN}Useful commands for application management:${NC}"
echo -e "  1. View running containers:"
echo -e "     ${YELLOW}docker ps${NC}"
echo -e "  2. Check live logs from the backend:"
echo -e "     ${YELLOW}cd $REPO_DIR/backend && docker compose logs -f${NC}"
echo -e "  3. Stop the application:"
echo -e "     ${YELLOW}cd $REPO_DIR/backend && docker compose down${NC}"
echo -e "     ${YELLOW}docker stop ptr-frontend${NC}"
echo -e "  4. Restart the stopped application:"
echo -e "     ${YELLOW}cd $REPO_DIR/backend && docker compose up -d${NC}"
echo -e "     ${YELLOW}docker start ptr-frontend${NC}"
echo -e ""
echo -e "${GREEN}==================================================================${NC}"