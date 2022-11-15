# AIRFluTe - An Intelligent Reading Fluency Teacher

![Alt text](https://github.com/HyperFlash123/AIRFluTe/blob/main/image.jpg)

**Children's Reading Assessment** is the task of evaluating the correctness of a child's read aloud speech of a passage. The main goal is to provide meaningful feedback of the read speech so that it can be used to improve their reading fluency. 

`AIRFluTe` is an end-to-end application that can perform the reading assessment task in real-time. It leverages the power of streaming deep learning models fine-tuned on children's speech to process audio and provide visual reading tracking and feedback.
This repository provides all the files needed to run the application locally on your browser.

**Prerequisites:** [Python](https://www.python.org/downloads/) and [Anaconda](https://docs.anaconda.com/anaconda/install/index.html).
## Getting Started

To install this application, run the following commands:
```
git clone https://github.com/HyperFlash123/AIRFluTe.git
cd AIRFlute
```
This will get a copy of the project installed locally.
To create a new conda environment, run the following command:
```
conda create --name <env>
conda activate <env>
pip install -r requirements.txt
```
You can then start the application on http://127.0.0.1:7860, by running:
```
python main.py
```

## License

Apache 2.0, see [LICENSE](LICENSE).
