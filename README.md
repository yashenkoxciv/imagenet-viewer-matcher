# imagenet-viewer-matcher

The service finds neighbors for encoded cards. Lists of neighbors are used to 
build distance matrix for clustering.

# Requirements

1. Ubuntu 20.04.3
2. Python 3.8.10
3. requirements.txt
4. python -m pip install git+https://github.com/yashenkoxciv/imagenet-viewer.git


# Expected environment variables

| Name               | Description                                                               |
|--------------------|:--------------------------------------------------------------------------|
| RABBITMQ_HOST      | RabbitMQ's host                                                           |
| INPUT_QUEUE        | RabbitMQ's queue with images to find matches                              |
| OUTPUT_QUEUE       | RabbitMQ's queue to push clustering request (to clusterizer)              |
| MONGODB_HOST       | MongoDB's connection string like this: mongodb://host:port/imagenetviewer |
| MILVUS_ALIAS       | Milvus's connection alias                                                 |
| MILVUS_HOST        | Milvus's server host                                                      |
| MILVUS_PORT        | Milvus's server port                                                      |
| MATCHING_THRESHOLD | Maximum distance to match images. Radius for KNN.                         |
| MATCHING_N         | Amount of images to build (send) clustering request.                      |

