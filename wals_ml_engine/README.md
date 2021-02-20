# Training your ML model using Google AI Platform and Custom Environment containers

This code implements a collaborative filtering recommendation model training using TensorFlow, Docker and Airflow, and applies
it to the simple user - ratings data set.

## Install

This code assumes python 2.

* Install miniconda2:

https://conda.io/docs/user-guide/install/index.html
```
$ sudo apt-get install -y git bzip2 unzip
```

```
wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
bash Miniconda2-latest-Linux-x86_64.sh
export PATH="/home/$USER/miniconda2/bin:$PATH"
```

* Create environment and install packages:

Assuming you are in the repo directory:

```
$ conda create -y -n tfrec
$ conda install -y -n tfrec --file conda.txt
$ source activate tfrec
$ pip install -r requirements.txt
```

* Install TensorFlow.

CPU:
```
$ pip install tensorflow==1.15
```

Or GPU, if one is available in your environment:

```
$ pip install tensorflow-gpu
```

## Versions used for custom environment
tensorflow==1.15
numpy==1.16.6
pandas==0.20.3
scipy==0.19.1


## Run

*   Train the model locally
```
$ ./mltrain.sh local ./data ratings_small.csv --data-type user_ratings
```


### How to build a Docker image for AI Platform custom environment
* Read: https://cloud.google.com/ai-platform/training/docs/custom-containers-training
* Read: https://cloud.google.com/ai-platform/training/docs/using-containers
* Read: https://cloud.google.com/sdk/gcloud/reference/ai-platform/jobs/submit/training#--master-image-uri
* Read: https://github.com/GoogleCloudPlatform/cloudml-samples/tree/master/tensorflow/containers/unsupported_runtime
* Read: https://cloud.google.com/ai-platform/training/docs/reference/rest/v1/projects.jobs#ReplicaConfig
* Read: https://airflow.readthedocs.io/_/downloads/en/1.10.2/pdf/

* 1. Run

```
export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
export IMAGE_REPO_NAME=recommendation_bespoke_container
export IMAGE_TAG=tf_rec_ur
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG
```
Run:
```
docker build -f Dockerfile -t $IMAGE_URI ./
```

* 2. Run
```
docker run $IMAGE_URI
```

* 3. Authenticate Docker
```
gcloud auth configure-docker
```
Push Docker image:
```
docker push $IMAGE_URI
```

The push refers to repository [gcr.io/<your-project>/recommendation_bespoke_container]

## Training in Google AI Platform

```
export BUCKET=<YOUR BUCKET NAME>
gsutil cp ./data/ratings_small.csv gs://$BUCKET/data/
./mltrain.sh train_custom gs://<YOUR BUCKET NAME> data/ratings_small.csv --data-type user_ratings
```

* 4. Update DAG:
Get the name of the Cloud Storage bucket created for you by Cloud Composer:
```
gcloud composer environments describe $CC_ENV \
    --location europe-west2 --format="csv[no-heading](config.dagGcsPrefix)" | sed 's/.\{5\}$//'
```
In the output, you see the location of the Cloud Storage bucket, like this:
gs://[region-environment_name-random_id-bucket]

Run:
```
export AIRFLOW_BUCKET="gs://[region-environment_name-random_id-bucket]"
gsutil cp ../airflow/dags/model_training.py ${AIRFLOW_BUCKET}/dags

```

## Docker tidy up unless you uesd -rm tag with build command

docker container prune --filter "until=24h"
docker image prune --filter "until=24h"
https://docs.docker.com/config/pruning/