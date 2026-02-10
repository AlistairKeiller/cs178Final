# Typst setup
All Typst work can be done via the web editor, the invite link is here: https://typst.app/project/wbHW9eVyJPrlDcLmOk4rRz
The Typst web editor does not have a version history in it's editor, so we should do backups to this repo just in case (for my peace of mind).
# Project setup
I initialized this repo with uv (this is a python package / version manager, an alternative to pip, venv, conda, etc. that is faster and more reproducible than the competition). Feel free to use those alternatives if you feel comfortable, but if we run into reproducabliltiy/version issues, I suggust trying uv.
## Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
## Install packages
```bash
uv sync
```
### Adding / removing packages
If u want to add a package (ml library, or anything else) to the project, just run
```bash
uv add PACKAGE_NAME
```
```bash
uv remove PACKAGE_NAME
```
## Editor
### Vscode
### Jupyter lab
```bash
uv run jupyter
```
### Jupyter notebook
```bash
uv run notebook
```
### VScode
This is what I will be using (technically less supported than jupyter, but I likeeeee it). Install the [jupyer extension pack](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter). Make sure when it prompts you for the environment, you select `.venv/bin/python`.