# Your MIDI files

Put `.mid` or `.midi` files in this folder (sub-folders are fine) and run `python preprocess.py`.
The model learns the style of whatever you give it: classical piano, jazz, game music...

- Use files you are allowed to use. Many sites offer free or public-domain MIDI collections, and Kaggle has several music datasets.
- 50 or more pieces of one style work better than a few pieces of many styles.
- If this folder is empty, the project falls back to the Bach chorales that ship with `music21`.
