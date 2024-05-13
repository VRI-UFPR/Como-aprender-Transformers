
# Possíveis Erros

Erro 1: TypeError: Unable to convert function return value to a Python type! The signature was () -> handle

```
pip3 install numpy --upgrade
```

Erro 2: AttributeError: module 'wandb.proto.wandb_internal_pb2' has no attribute 'Result'

```
pip3 install wandb --upgrade
```


Erro 3: OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a Python package or a valid path to a data directory.

```
python3 -m spacy download en_core_web_sm
```

Erro: externally-managed-environment

```
sudo rm /usr/lib/python3.11/EXTERNALLY-MANAGED
```