# Usage

## Preparation

1. Create a new python virtual environment.
   Inference in incompatible with Python 3.11 and higher, so I recommend using Python 3.10 with pyenv.

```shell
pyenv install 3.10.5
pyenv global 3.10.5
virtualenv venv_name
source venv_name/Scripts/activate
```

2. Clone repository and install requirements.

```shell
git clone https://github.com/jastinjn/bus-lane-intrusion.git
cd bus-lane-intrusion
pip install -r requirements.txt
```

## Using the model

Run the model on image files (.png, .jpg) and video files (.mp4). Several examples are included under images/ and videos/ to try.

```shell
python main.py path/to/image/video
```
