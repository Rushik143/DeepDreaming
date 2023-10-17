import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def deep_dream(model, image, layer_name, iterations=20, step_size=0.01, octave_scale=1.5, num_octaves=3):
    img = tf.keras.preprocessing.image.img_to_array(image)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)

    # Reshape the input image to match the expected shape of the model
    img = np.expand_dims(img, axis=0)

    # Create a model that outputs the activation of the specified layer
    layer_output = model.get_layer(layer_name).output
    dream_model = tf.keras.Model(model.input, layer_output)

    # Generate the DeepDream image
    original_shape = img.shape[1:-1]
    img = tf.image.resize(img, original_shape)

    for _ in range(num_octaves):
        # Generate details from the upsampled image
        img_details = tf.Variable(img)
        for _ in range(iterations):
            with tf.GradientTape() as tape:
                tape.watch(img_details)
                outputs = dream_model(img_details)
                loss = tf.reduce_mean(outputs)

            gradients = tape.gradient(loss, img_details)
            gradients /= tf.math.reduce_std(gradients) + 1e-8
            img_details = tf.Variable(img_details + gradients * step_size)

            img_details = tf.clip_by_value(img_details, -1, 1)

        # Upscale the image for the next octave
        img = tf.image.resize(img_details, original_shape)
        step_size *= octave_scale

    img = tf.clip_by_value(img, -1, 1)
    img = tf.keras.preprocessing.image.array_to_img(img[0])
    return img
