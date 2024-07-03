# Dual-ResShift: Dual-input Separated Features Residual Shift Diffusion Model for CTA Image Super-resolution

## Brief
This is an implementation of Dual-ResShift by PyTorch. Thank you for your reading! We release processed test data and trained models you can test with our data and code. The code for the data processing and training parts will be released after our paper is accepted. Our code is based on [This](https://github.com/zsyOAOA/ResShift)
<br />Model download address: [mode](https://drive.google.com/drive/folders/109UfiqeiBwjB-VopWsTDH9SA5GnJI-K8?usp=drive_link)
<br />Test data download address: [test data](https://drive.google.com/drive/folders/1KeWI0IjmUjysVIyzs-mSOnZVUjQxgmZk?usp=drive_link)
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
python dual_inference.py -o result/path -m model/path
```
