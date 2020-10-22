import numpy as np
from tqdm import tqdm

def eval(val_loader, model, criterion, params, epoch):
    """Main method to evaluate model."""
    model.eval()
    print("Evaluating...")

    loss = 0
    num_classes = params["num_classes"]

    confusion = np.zeros((num_classes, num_classes))
    #print(confusion)
    for iteration, (steps, targets, _) in enumerate(tqdm(val_loader)):
        if params["use_cuda"]:
            steps = steps.cuda()
            targets = targets.cuda()
        # print(steps)
        output = model(steps)

        rows = targets.cpu().numpy()
        cols = output.max(1)[1].cpu().numpy()
        # print(output,output.max(1),output.max(1)[1])
        # print(rows,cols)
        # print(confusion[rows, cols])
        confusion[rows, cols] += 1
        # print(confusion[rows, cols])
        # print(confusion)
        loss += criterion(output, targets)

    loss = loss / iteration
    acc = np.trace(confusion) / np.sum(confusion)
    print(np.trace(confusion))
    #Plot confusion matrix in visdom
    # logger.heatmap(confusion, win='4', opts=dict(
    #     title="Confusion_Matrix_epoch_{}".format(epoch),
    #     columnnames=["A","B","C"],
    #     rownames=["A","B","C"])
    # )

    return loss, acc