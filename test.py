import torch
import visdom
import argparse
import os
import pandas as pd

from config.set_params import params as sp
from modeling.model import HARmodel
from utils.preprocessing import HARdataset

def build_testset(params):
    df = pd.read_csv(params["test"], low_memory=False)
    #parts = ["belt", "arm", "dumbbell", "forearm"]
    self.variables = ["fIAV_muscle1", "fLogD_muscle1", "fMAV_muscle1", "fMAX_muscle1", "fNZM_muscle1", "fRMS_muscle1", "fSSC_muscle1", "fVAR_muscle1", "fWA_muscle1", "fWL_muscle1", "fZC_muscle1",
                      "fARC_muscle1", "fFME_muscle1", "fFMD_muscle1", "fIAV_muscle2", "fLogD_muscle2", "fMAV_muscle2", "fMAX_muscle2", "fNZM_muscle2", "fRMS_muscle2", "fSSC_muscle2", "fVAR_muscle2",
                      "fWA_muscle2", "fWL_muscle2", "fZC_muscle2", "fARC_muscle2", "fFME_muscle2", "fFMD_muscle2", "fIAV_muscle3", "fLogD_muscle3", "fMAV_muscle3", "fMAX_muscle3", "fNZM_muscle3",
                      "fRMS_muscle3", "fSSC_muscle3", "fVAR_muscle3", "fWA_muscle3", "fWL_muscle3", "fZC_muscle3", "fARC_muscle3", "fFME_muscle3", "fFMD_muscle3", "fIAV_muscle4", "fLogD_muscle4",
                      "fMAV_muscle4", "fMAX_muscle4", "fNZM_muscle4", "fRMS_muscle4", "fSSC_muscle4", "fVAR_muscle4", "fWA_muscle4", "fWL_muscle4", "fZC_muscle4", "fARC_muscle4", "fFME_muscle4",
                      "fFMD_muscle4", "fDisp1_glove", "fDisp2_glove", "fDisp3_glove", "fDisp4_glove", "fDisp5_glove", "fDisp6_glove", "fDisp7_glove", "fDisp8_glove", "fDisp9_glove", "fDisp10_glove",
                      "fDisp11_glove", "fDisp12_glove", "fDisp13_glove", "fDisp14_glove", "fDisp_Hem1", "fDisp_Hem2", "fDisp_Dem12", "fDisp_mfce", "fDisp_Tposx", "fDisp_Tposy", "fDisp_angle"]

    var_list = []
    #for part in parts:
    for var in variables:
        var_list.append(list(df[var]))#.format(part)
    var_list = torch.tensor(var_list)

    return var_list

def main():
    """Driver file to run inference on test data."""
    className = ["S","U"]

    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
    )
    args = parser.parse_args()
    assert type(args.checkpoint) is str, "Please input path to checkpoint"

    params = sp().params

    model = HARmodel(params["input_dim"], params["num_classes"])

    if os.path.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["state_dict"])
        print("=> loaded checkpoint '{}'".format(args.checkpoint))

    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))
        return

    dataset = HARdataset(params["root"])
    mean, std = dataset.mean, dataset.std

    logger = visdom.Visdom()

    testset = build_testset(params)
    testset = (testset - mean) / std

    results = []
    for i in range(testset.size(1)):
        test_data = testset[:,i].view(1, 1, -1)
        output = model(test_data)
        results.append(int(output.max(1)[1]))

    results = [className[i] for i in results]
    print("Prediction results:")
    print(results)

if __name__ == "__main__":
    main()

