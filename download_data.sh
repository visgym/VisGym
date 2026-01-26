mkdir -p data/

# 5 options:
# DEFAULT_IMAGE, DEFAULT_VIDEO, DEFAULT_3D, DEFAULT_REFCOCO, DEFAULT_LVIS
# DEFAULT_3D will only download the first 2 directories (000-000 to 000-001) for size consideration
# if you want to download all directories, change the END_NUM in the script to 159 and start a background job to download it.

if [ $1 == "DEFAULT_IMAGE" ]; then
    mkdir -p data/images
    # download https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip
    wget -O data/images/llava_training_set.zip https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip
    unzip data/images/llava_training_set.zip -d data/images
    rm data/images/llava_training_set.zip
fi

if [ $1 == "DEFAULT_VIDEO" ]; then
    mkdir -p data/videos
    cd data/videos
    
    # Download Something-Something-V2 dataset parts
    echo "Downloading Something-Something-V2 dataset parts..."
    wget -O 20bn-something-something-v2-00 "https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-v2-00"
    wget -O 20bn-something-something-v2-01 "https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-v2-01"
    
    # Download labels (this one IS actually a zip file)
    echo "Downloading labels..."
    wget -O 20bn-something-something-download-package-labels.zip "https://softwarecenter.qualcomm.com/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-download-package-labels.zip"
    unzip 20bn-something-something-download-package-labels.zip
    rm 20bn-something-something-download-package-labels.zip
    
    # Concatenate and extract video files from the tar parts
    echo "Extracting video files from concatenated tar archive..."
    cat 20bn-something-something-v2-?? | tar -xvzf -
    
    # Clean up tar parts
    echo "Cleaning up temporary files..."
    rm 20bn-something-something-v2-??
    
    cd ../..
fi

if [ $1 == "DEFAULT_3D" ]; then
    mkdir -p data/3d
    cd data/3d
    
    # Download Objaverse glbs directories 000-000 to 000-XXX
    # Change END_NUM to 159 to download all directories (000-000 to 000-159)
    END_NUM=2
    
    echo "Downloading Objaverse 3D models from 000-000 to 000-$(printf "%03d" $END_NUM)..."
    
    # Using hf CLI (install with standalone installer)
    if ! command -v hf &> /dev/null; then
        echo "Error: hf CLI not found. Please install it with:"
        echo "  curl -LsSf https://hf.co/cli/install.sh | bash"
        exit 1
    fi
    
    echo "Using hf CLI..."
    INCLUDE_ARGS=""
    for i in $(seq 0 $END_NUM); do
        DIR_NAME=$(printf "000-%03d" $i)
        INCLUDE_ARGS="$INCLUDE_ARGS --include glbs/$DIR_NAME/*"
    done
    hf download allenai/objaverse $INCLUDE_ARGS --repo-type dataset --local-dir .
    
    cd ../..
fi

if [ $1 == "DEFAULT_REFCOCO" ]; then
    mkdir -p data/refcoco
    cd data/refcoco
    
    # Download refcoco+ dataset
    echo "Downloading refcoco+ dataset..."
    wget -O refcoco+.zip https://web.archive.org/web/20220413011656/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip
    unzip refcoco+.zip
    rm refcoco+.zip
    
    Download MSCOCO train2014 images
    echo "Downloading MSCOCO train2014 images..."
    mkdir -p images/mscoco/images
    cd images/mscoco/images
    wget -O train2014.zip http://images.cocodataset.org/zips/train2014.zip
    unzip train2014.zip
    rm train2014.zip
    
    cd ../../..
    cd ../..
fi

if [ $1 == "DEFAULT_LVIS" ]; then
    mkdir -p data/lvis
    cd data/lvis
    
    # Download LVIS v1 train annotations
    echo "Downloading LVIS v1 train annotations..."
    wget -O lvis_v1_train.json.zip https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip
    unzip lvis_v1_train.json.zip
    rm lvis_v1_train.json.zip
    
    # Download COCO train2017 images
    echo "Downloading COCO train2017 images..."
    wget -O train2017.zip http://images.cocodataset.org/zips/train2017.zip
    unzip train2017.zip
    rm train2017.zip
    
    cd ../..
fi

# if all
if [ $1 == "ALL" ]; then
    mkdir -p data/images
    mkdir -p data/videos
    mkdir -p data/3d
    mkdir -p data/refcoco
    mkdir -p data/lvis
    download_data.sh DEFAULT_IMAGE
    download_data.sh DEFAULT_VIDEO
    download_data.sh DEFAULT_3D
    download_data.sh DEFAULT_REFCOCO
    download_data.sh DEFAULT_LVIS
fi
