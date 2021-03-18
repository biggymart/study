# 이미지 유사도 탐지기
# https://medium.com/daangn/%EC%9D%B4%EB%AF%B8%EC%A7%80-%ED%83%90%EC%A7%80%EA%B8%B0-%EC%89%BD%EA%B2%8C-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0-abd967638c8e
# 이미지만으로 카테고리 분류
# https://medium.com/daangn/%EC%9D%B4%EB%AF%B8%EC%A7%80%EB%A7%8C%EC%9C%BC%EB%A1%9C-%EB%82%B4-%EC%A4%91%EA%B3%A0%EB%AC%BC%ED%92%88%EC%9D%98-%EC%B9%B4%ED%85%8C%EA%B3%A0%EB%A6%AC%EB%A5%BC-%EB%B6%84%EB%A5%98%ED%95%B4%EC%A4%80%EB%A9%B4-feat-keras-b86e5f286c71

import tensorflow_hub as hub
module = hub.Module("https://tfhub.dev/google/imagenet/...")

outputs = module(dict(images=daangn_profile_image), signature="image_feature_vector", as_dict=True)
target_image = outputs['default']

outputs = module(dict(images=user_profile_images), signature="image_feature_vector", as_dict=True)
input_image = outputs['default']

dot = tf.tensordot(target_image, tf.transpose(input_image), 1)
similarity = dot / (tf.norm(target_image, axis=1) * tf.norm(input_image, axis=1))
similarity = tf.reshape(similarity, [-1])