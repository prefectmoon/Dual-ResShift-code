# Dual-ResShift: Daul-input Separated Features Residual Shift Diffusion Model for CTA Image Super-resolution

## Brief

This is an implementation of Dual-ResShift by PyTorch. Thank you for your reading! We release processed test data and trained models that you can test with our data and code. The code of the data processing part and the training part will be released after our paper is accepted. (Our code is based on [This](https://github.com/zsyOAOA/ResShift)

## Usage

### Environment

<code>
pip install -r requirement.txt
</code>

### Data Prepare


### Training


### Test
```
#Set the path to the test data in config/train.yaml
cd Dual-Resshift-code
python daul_inference.py -o result/path -m model/path
```
