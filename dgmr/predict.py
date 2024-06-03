import tensorflow as tf
import tensorflow_hub as hub
from einops import rearrange

from dgmr.settings import MODEL_PATH


def load_deepmind_model(image_size: tuple):
    """
    Load the deepmind pre-trained model from a google storage (with tensorflow_hub).
    """
    print("--> Loading model...")
    available_models = [(1536, 1280)]
    if image_size not in available_models:
        raise Exception(
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


def predict_deepmind(
    X: tf.Tensor,
    num_members=1,
):
    """Make predictions from a TF-Hub snapshot of the 'Generative Method' model.

    Args:
      input_frames: Shape (T_in, H, W, C), where T_in = 4. Input frames to condition
        the predictions on.
      num_members: The number of different samples to draw.

    Returns:
      A tensor of shape (num_members, T_out, H, W, C), where T_out is either 18 or 22
      as described above.
    """
    image_size = X.shape[1:3]
    model = load_deepmind_model(image_size)

    X = tf.math.maximum(X, 0.0)
    # Add a batch dimension and tile along it to create a copy of the input for
    # each sample:
    X = tf.expand_dims(X, 0)
    X = tf.tile(X, multiples=[num_members, 1, 1, 1, 1])

    # Sample the latent vector z for each sample:
    _, input_signature = model.structured_input_signature
    z_size = input_signature["z"].shape[1]
    # Constant perturbation to always keep the same model :
    z_samples = tf.random.normal(shape=(num_members, z_size)) * 0

    inputs = {
        "z": z_samples,
        "labels$onehot": tf.ones(shape=(num_members, 1)),
        "labels$cond_frames": X,
    }
    print("--> Predict....")
    samples = model(**inputs)["default"]
    samples = samples[:, 4:, ...]
    # will return a total of 22 frames
    # along the time axis, the 4 input frames followed by 18 predicted frames.
    # Take positive values of rainfall only.
    samples = tf.math.maximum(samples, 0.0)

    samples = rearrange(samples, "b t h w c -> b c t h w")
    return samples


if __name__ == "__main__":
    model = load_deepmind_model((1536, 1280))
    print(model)
