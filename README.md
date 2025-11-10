# Introduction

![Running SLMs in web browsers](docs/thumb_small_language_model.jpg)

This repository is part of [playbook for experiments on fine tuning small language models](https://ashishware.com/2025/11/16/slm_in_browser/) using LoRA, exporting them to ONNX and running them locally using ONNX compatibale runtime  like javascript(node js) and WASM (browser)

### Before you start

- Clone the repository.
- From https://www.kaggle.com/code/finalepoch/smollm-360-lora-onnx-inference/output?scriptVersionId=274675806 , copy `model.onnx` and `model_q8.onnx` to the `model_files` directory of the repository.
- Run `npm install`

### To run NodeJS example (NodeJS + onnx-runtime, server side)

- Simple run `node app.js`

This is what you should see

![NodeJS application showing paraphrasing screen](docs/slm_nodejs.gif)


### To run web browser based demo (WASM based in-browser inference)

- Simply access `web.html` from a local server (example `http://localhost:3000/web.html`)

This is what you should see 

![NodeJS application showing paraphrasing screen](docs/slm_web_wasm.gif)


### Citation

```
@misc{allal2024SmolLM,
      title={SmolLM - blazingly fast and remarkably powerful}, 
      author={Loubna Ben Allal and Anton Lozhkov and Elie Bakouch and Leandro von Werra and Thomas Wolf},
      year={2024},
}
```
