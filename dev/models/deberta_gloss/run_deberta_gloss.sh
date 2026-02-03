#!/bin/bash

# Nastavení řídících parametrů plánovače Slurm
#SBATCH --job-name=wsd_deberta_gloss
#SBATCH --output="logs/wsd_deberta_gloss-%J.out"
#SBATCH --error="logs/wsd_deberta_gloss-%J.err"

# Název python virtuálního prostředí
PYTHON_ENVIRONMENT_NAME="wsd_env"
PYTHON_PROJECT_NAME="mvi"
PYTHON="python3.11"

#------------------------------------------------------------------------------

NODE_HOSTNAME=$(hostname)
PYTHON_PROJECT_DIR="${HOME}/${PYTHON_PROJECT_NAME}"
PYTHON_VIRTUAL_ENVIRONMENT="${PYTHON_PROJECT_DIR}/${PYTHON_ENVIRONMENT_NAME}"

PYTHON_PACKAGES="pip pandas numpy scikit-learn torch transformers datasets evaluate accelerate sentencepiece pyarrow protobuf nltk"

###############################################################################

cd "${PYTHON_PROJECT_DIR}"

if [ ! -d "${PYTHON_VIRTUAL_ENVIRONMENT}" ]
then
    echo "Vytvářím python virtuální prostředí: ${PYTHON_VIRTUAL_ENVIRONMENT}"
    ${PYTHON} -m venv "${PYTHON_VIRTUAL_ENVIRONMENT}"
fi

echo "Aktivuji python virtuální prostředí: ${PYTHON_VIRTUAL_ENVIRONMENT}"
source "${PYTHON_VIRTUAL_ENVIRONMENT}/bin/activate"

echo "Aktualizuji/Instaluji potřebné balíčky..."
pip install --upgrade ${PYTHON_PACKAGES}

echo ""
echo "SPOUŠTÍM TRÉNOVÁNÍ (GLOSS) NA UZLU: ${NODE_HOSTNAME}"
echo "GPU info:"
nvidia-smi
echo ""

# Running the new gloss training script
python train_deberta_gloss.py \
    --data_file "semcor_gloss_train.parquet" \
    --output_dir "./deberta_wsd_gloss" \
    --model_name "microsoft/deberta-v3-base" \
    --epochs 3 \
    --batch_size 16 \
    --lr 2e-5

echo ""
echo "Trénování bylo dokončeno."
echo ""
