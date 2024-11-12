# AFFPD-ONEE: Overlapping and Nested Event Extraction Based on Adaptive Fusion Filtering and Position Decoding


## 1. Environments

```
- python (3.8.12)
- cuda (11.4)
```

## 2. Dependencies

```
- numpy (1.19.2)
- torch (1.10.0)
- transformers (4.10.0)
- prettytable (2.1.0)
```

## 3. Dataset

- FewFC: Chinese Financial Event Extraction dataset. The original dataset can be accessed at [this repo](https://github.com/TimeBurningFish/FewFC). Here we follow the settings of [CasEE](https://github.com/JiaweiSheng/CasEE). Note that the data is avaliable at /data/fewFC, and we adjust data format for simplicity of data loader. To run the code on other dataset, you could also adjust the data as the data format presented.
- [ge11: GENIA Event Extraction (GENIA), 2011](https://2011.bionlp-st.org/home/genia-event-extraction-genia)

## 4. Preparation

- Download dataset
- Process them to fit the same format as the example in `data/`
- Put the processed data into the directory `data/`

## 5. Training

```bash
>> python main.py --config ./config/fewfc.json
```

