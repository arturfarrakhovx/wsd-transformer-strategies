#!/bin/bash

# Nastavení řídících parametrů plánovače Slurm
#SBATCH --job-name=wsd_deberta_train
#SBATCH --output="logs/wsd_deberta_train-%J.out"
#SBATCH --error="logs/wsd_deberta_train-%J.err"

# Název python virtuálního prostředí
PYTHON_ENVIRONMENT_NAME="wsd_env"

# Název adresáře, obsahující python projekt
PYTHON_PROJECT_NAME="mvi"

# Která verze z dostupných verzí pythonu se má použít
PYTHON="python3.11"

#------------------------------------------------------------------------------

# Na jakém výpočetním uzlu se spustí úloha
NODE_HOSTNAME=$(hostname)

# Cesta k python projektu
PYTHON_PROJECT_DIR="${HOME}/${PYTHON_PROJECT_NAME}"

# Cesta k python virtuálnímu prostředí
PYTHON_VIRTUAL_ENVIRONMENT="${PYTHON_PROJECT_DIR}/${PYTHON_ENVIRONMENT_NAME}"

# Seznam balíčků k instalaci pro WSD úlohu
# Zahrnuje transformers, torch, sentencepiece, protobuf (nutné pro DeBERTa V3)
PYTHON_PACKAGES="pip pandas numpy scikit-learn torch transformers datasets evaluate accelerate sentencepiece pyarrow protobuf"

###############################################################################

# Přejdu do adresáře s python projektem (pokud tam nejsem)
cd "${PYTHON_PROJECT_DIR}"

if [ ! -d "${PYTHON_VIRTUAL_ENVIRONMENT}" ]
then
    echo "Vytvářím python virtuální prostředí: ${PYTHON_VIRTUAL_ENVIRONMENT}"
    ${PYTHON} -m venv "${PYTHON_VIRTUAL_ENVIRONMENT}"
fi

echo "Aktivuji python virtuální prostředí: ${PYTHON_VIRTUAL_ENVIRONMENT}"
source "${PYTHON_VIRTUAL_ENVIRONMENT}/bin/activate"

echo "Aktualizuji/Instaluji potřebné balíčky..."
# Instalace všech balíčků najednou
pip install --upgrade ${PYTHON_PACKAGES}

echo ""
echo "SPOUŠTÍM TRÉNOVÁNÍ NA UZLU: ${NODE_HOSTNAME}"
echo "GPU info:"
nvidia-smi
echo ""

# Spuštění Python skriptu
# Soubory 'semcor_train.parquet' a 'label_map.json' musí být ve stejné složce
python train_deberta.py \
    --data_file "semcor_train.parquet" \
    --label_map "label_map.json" \
    --output_dir "./deberta_wsd_custom" \
    --model_name "microsoft/deberta-v3-base" \
    --epochs 5 \
    --batch_size 16 \
    --grad_accum 1 \
    --lr 5e-5

echo ""
echo "Trénování bylo dokončeno."
echo ""
