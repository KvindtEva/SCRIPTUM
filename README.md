Trash:
Google drive: https://drive.google.com/drive/folders/1s5ihCQ-0R_EnJlkUd_tvDw6hp6ybcpyb 
 
GitHub of EuPaC: https://github.com/CCS-ZCU/EuPaC_shared/tree/master

Access to dataset: https://huggingface.co/datasets/janko/250521-scriptum


# SCRIPTUM Project

(description of project)

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- On macOS/Linux:
```bash
source venv/bin/activate
```
- On Windows:
```bash
.\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure


## Limitations regarding the Data

- The Meta-dataset contains over 11.600 datapoints, where as the datasets with the fulltext of the journals only contains around 10.500 datapoints.
- The ocr (optical character recognition) of the journals contains errors that we werent able to correct.
- For the journals there are some (how many?) where the publication year was missing in the meta-data. We were able to add a publication year to some, but not for all of them. We exluded these of our analysis.