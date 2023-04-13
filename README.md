# pengbot-aligner
Tool to align images in Penguin Watch's penguin counting tool

## Installation
Clone the git repository
```
git clone https://github.com/CarlosArteta/pengbot-aligner.git
```
 
Create a Python virtual enviroment
```
cd pengbot-aligner
python3 -m venv venv_pengbot_aligner
```

Install the package
```
source venv_pengbot_aligner/source/bin/activate
pip install .
```

(Optional) Run tests
```
pytest 
```

## Usage
Activate the python virtual environment 
```
source venv_pengbot_aligner/source/bin/activate
```

Edit a configuration file; for example: `pengbot-aligner/example/config_AITCb2014a.yaml`

Call the tool with the edited config file:
```
pengbot-aligner --config example/config_AITCb2014a.yaml
```





