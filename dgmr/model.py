import tensorflow as tf
import tensorflow_hub as hub

from dgmr.settings import MODEL_PATH


def load_model(image_size: tuple):
    """
    Load the DGMR pre-trained model from a google storage (with tensorflow_hub).
    """
    print("--> Loading model...")
    available_models = [(1536, 1280)]
    if image_size not in available_models:
        raise ValueError(
            f"Height and width are not available for this model, choose from {image_size}"
        )

    hub_module = hub.load(
        str(MODEL_PATH / f"deepmind_{image_size[0]}_{image_size[1]}/")
    )
    # Note this has loaded a legacy TF1 model for running under TF2 eager mode.
    # This means we need to access the module via the "signatures" attribute. See
    # https://github.com/tensorflow/hub/blob/master/docs/migration_tf2.md#using-lower-level-apis
    # for more information.
    model = hub_module.signatures["default"]

    return model


def predict(x: tf.Tensor, num_members=1):
    """Makes a prediction from the DGMR TF-Hub snapshot.

    Args:
      input_frames: Shape (T_in, H, W, C), where T_in = 4. Input frames to condition
        the predictions on.
      num_members: The number of different samples to draw.

    Returns:
      A tensor of shape (num_members, T_in+T_out, H, W), where T_out=18.
    """
    image_size = x.shape[1:3]
    model = load_model(image_size)

    x = tf.expand_dims(x, 0)  # Add batch dimension

    # Tile along batch dim to create a copy of the input for each member
    x = tf.tile(x, multiples=[num_members, 1, 1, 1, 1])

    # Sample the latent vector z for each member
    _, input_signature = model.structured_input_signature
    z_size = input_signature["z"].shape[1]
    z_samples = tf.random.normal(shape=(num_members, z_size))
    if num_members == 1:  # Constant perturbation when only 1 member
        z_samples = z_samples * 0

    inputs = {
        "z": z_samples,
        "labels_onehot": tf.ones(shape=(num_members, 1)),
        "labels_cond_frames": x,
    }

    print("--> Predict....")
    output = model(**inputs)["default"]  # returns input + output frames = 22 frames
    return output[:, :, :, :, 0]  # remove channel dims
