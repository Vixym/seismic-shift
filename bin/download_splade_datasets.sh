# Inspired by a similar data downloading script:
# https://github.com/BlaiseMuhirwa/flatnav/blob/main/bin/download_ann_benchmarks_datasets.sh


# Create a list of benchmark datasets to download.
BENCHMARK_DATASETS=("msmarco-splade"
		    "msmarco-effsplade"
		    "msmarco-unicoilT5"
		    "nq-splade")


function print_help() {
    echo "Usage: ./download_splade_datasets.sh <dataset>"
    echo ""
    echo "Available datasets:"
    echo "${BENCHMARK_DATASETS[@]}"
}

function download_dataset() {
    local dataset=$1

    # Skip download if directory data/dataset_name already exists.
    if [ -d "data/${dataset}" ]; then
        echo "data/${dataset} already exists. Skipping download."
        exit 0
    fi

    mkdir -pv data/${dataset}
    cd data/${dataset}

    for file_type in documents queries
      do
        url="https://huggingface.co/datasets/tuskanny/seismic-${dataset}/resolve/main/${file_type}.tar.gz?download=true"
        wget ${url} -O ${file_type}.tar.gz && tar -xvzf ${file_type}.tar.gz
      done
}

# If the first argument is -h or --help, then print help and exit.
if [[ $1 == "-h" || $1 == "--help" ]]; then
    print_help
    exit 0
fi

download_dataset $1
