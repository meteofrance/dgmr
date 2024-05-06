import time

from flask import Flask, render_template, send_file
from settings import PLOT_PATH, PROD_PATH, SERVER_PORT

app = Flask(__name__)


def read_with_lock(path_file, mimetype):
    "Reads a file and avoids collision btw reader and writer by using a lock."
    lock = PROD_PATH / "lock_file.txt"
    for _ in range(5):
        try:
            # Attempt to acquire a lock
            with open(lock, "x"):
                img = send_file(path_file, mimetype=mimetype)
                lock.unlink()  # Release the lock
                return img
        except FileExistsError:
            print(f"Another process is writing {path_file}. Retrying in a moment...")
            time.sleep(1)


@app.route("/last_forecast")
def view_last_forecast():
    return read_with_lock(PLOT_PATH / "last_forecast.gif", "image/gif")


@app.route("/last_gif")
def view_last_gif():
    return read_with_lock(PLOT_PATH / "last_gif.gif", "image/gif")


@app.route("/last_error")
def view_last_image():
    return read_with_lock(PLOT_PATH / "last_error.png", "image/png")


@app.route("/")
def list_files():
    gif_files = sorted(PLOT_PATH.glob("*.gif"), reverse=True)
    gif_files = [f.name for f in gif_files]
    png_files = sorted(PLOT_PATH.glob("*.png"), reverse=True)
    png_files = [f.name for f in png_files]
    return render_template("file_list.html", gif_files=gif_files, png_files=png_files)


@app.route("/view_gif/<filename>")
def view_gif(filename):
    image_path = PLOT_PATH / filename
    return send_file(image_path, "image/gif")


@app.route("/view_image/<filename>")
def view_image(filename):
    image_path = PLOT_PATH / filename
    return send_file(image_path, "image/png")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=SERVER_PORT)  # nosec B104
