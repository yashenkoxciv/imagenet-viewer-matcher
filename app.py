import os
import uuid
import pika
import logging
import numpy as np
from bson import ObjectId
from environs import Env
from mongoengine import connect, disconnect
from imagenetviewer.image import Image, Neighbor, ImageStatus
from imagenetviewer.vector import ImagesFeatures, ImagesMatching
from pymilvus import connections, utility


def on_request(ch, method, props, body):
    image_object_id = ObjectId(body.decode())
    image = Image.objects.get(pk=image_object_id)

    # sometimes throws IndexError: list index out of range
    # because of Milvus's consistency_level
    # you have to wait some time (1) to do .get_vector,
    # or explicitly use consistency_level on this query (2)
    # (1) https://blog.rabbitmq.com/posts/2015/04/scheduling-messages-with-rabbitmq
    # (2) https://milvus.io/docs/v2.0.x/comparison.md#Tunable-consistency
    z = np.array(imgf.get_vector(image.vector_id)[0]['features'])
    # z = np.array(
    #     imgf.collection.query(
    #         expr=f'pk in [{image.vector_id}]',
    #         output_fields=['features'],
    #         consistency_level='Strong'
    #     )
    # )

    # 1) do similarity search
    global imgm
    search_result = imgm.search([z], k=1000)
    matches = search_result[0]

    # 2) remember neighbors for the image
    matched_num = 0
    for match in matches:
        if match.distance < env.float('MATCHING_THRESHOLD'):
            matched_image = Image.objects.get(vector_id=match.id)

            neighbor = Neighbor(matched_image=matched_image, distance=match.distance)
            image.neighbors.append(neighbor)

            matched_num += 1
    logger.debug(f'matched {image_object_id} with {matched_num} images')

    # 3) push vector (z) to database
    mr = imgm.insert_vectors_with_pk([image.vector_id], [z])

    # 4) put clustering_request_id to image object
    global current_clustering_id
    image.clustering_request_id = current_clustering_id
    image.status = ImageStatus.PENDING_CLUSTERING

    image.save()

    global request_count
    request_count += 1

    if request_count >= env.int('MATCHING_N'):
        # 1) drop vector collection
        utility.drop_collection(ImagesMatching.collection_name)

        imgm = ImagesMatching()
        imgm.collection.load()

        # send clustering_request_id to clusterizer
        ch.basic_publish(
            exchange='',
            routing_key=env('OUTPUT_QUEUE'),
            body=str(current_clustering_id)
        )

        logger.info(f'clustering request: {current_clustering_id} sent')

        current_clustering_id = str(uuid.uuid4())

        request_count = 0

    ch.basic_ack(delivery_tag=method.delivery_tag)



if __name__ == '__main__':
    logger = logging.getLogger('imagenet-matcher')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(asctime)s %(name)s %(levelname)s] %(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    env = Env()
    env.read_env()

    connect(host=env('MONGODB_HOST'), uuidRepresentation='standard')

    connections.connect(env('MILVUS_ALIAS'), host=env('MILVUS_HOST'), port=env('MILVUS_PORT'))
    imgf = ImagesFeatures()
    imgf.collection.load()
    imgm = ImagesMatching()
    imgm.collection.load()

    con_par = pika.ConnectionParameters(
        heartbeat=600,
        blocked_connection_timeout=300,
        host=env('RABBITMQ_HOST')
    )
    connection = pika.BlockingConnection(con_par)
    channel = connection.channel()

    channel.queue_declare(queue=env('INPUT_QUEUE'), durable=True)
    channel.queue_declare(queue=env('OUTPUT_QUEUE'), durable=True)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=env('INPUT_QUEUE'), on_message_callback=on_request)

    current_clustering_id = str(uuid.uuid4())
    request_count = 0

    logger.info('[+] awaiting images to recognize')
    channel.start_consuming()




